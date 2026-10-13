import heapq
import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import BinaryIO

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096 

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  
        while True:
            mini_chunk = file.read(mini_chunk_size)  

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))

def iter_pairs(tokens):
    for i in range(len(tokens) - 1):
        yield (tokens[i], tokens[i + 1])

def build_pair_indexes(word_id_to_tokens, word_id_to_count):
    pair_counts = {}
    pair_to_word_ids = defaultdict(set)
    for word_id, tokens in word_id_to_tokens.items():
        count = word_id_to_count[word_id]
        for pair in iter_pairs(tokens):
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_to_word_ids[pair].add(word_id)
    return pair_counts, pair_to_word_ids

class _ReversePair:
    __slots__ = ("pair",)

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair

    def __eq__(self, other):
        return isinstance(other, _ReversePair) and self.pair == other.pair


class _PairMaxHeap:
    """Lazy max-heap matching max((count, pair)) ordering exactly."""

    def __init__(self, pair_counts):
        self.heap = [(-count, _ReversePair(pair), pair) for pair, count in pair_counts.items()]
        heapq.heapify(self.heap)

    def update(self, pair, count):
        if count > 0:
            heapq.heappush(self.heap, (-count, _ReversePair(pair), pair))

    def pop_max(self, pair_counts):
        while self.heap:
            negative_count, _, pair = heapq.heappop(self.heap)
            if pair_counts.get(pair) == -negative_count:
                return pair
        raise KeyError("no remaining pair in priority heap")

    def compact_if_needed(self, pair_counts):
        if len(self.heap) > max(64, 2 * len(pair_counts)):
            self.heap = [(-count, _ReversePair(pair), pair) for pair, count in pair_counts.items()]
            heapq.heapify(self.heap)


def apply_merge_incremental(best_pair, word_id_to_tokens, word_id_to_count, pair_counts, pair_to_word_ids):
    """Merge one pair and return the pair counts changed by local deltas."""
    affected_word_ids = tuple(pair_to_word_ids.get(best_pair, ()))
    pair_deltas = defaultdict(int)

    for word_id in affected_word_ids:
        tokens = word_id_to_tokens[word_id]
        count = word_id_to_count[word_id]
        new_tokens = merge_one_token_sequence(tokens, best_pair)
        old_pairs = Counter(iter_pairs(tokens))
        new_pairs = Counter(iter_pairs(new_tokens))

        for pair, old_occurrences in old_pairs.items():
            delta = (new_pairs.get(pair, 0) - old_occurrences) * count
            if delta:
                pair_deltas[pair] += delta
        for pair, new_occurrences in new_pairs.items():
            if pair not in old_pairs:
                pair_deltas[pair] += new_occurrences * count

        for pair in old_pairs.keys() - new_pairs.keys():
            word_ids = pair_to_word_ids[pair]
            word_ids.remove(word_id)
            if not word_ids:
                del pair_to_word_ids[pair]
        for pair in new_pairs.keys() - old_pairs.keys():
            pair_to_word_ids[pair].add(word_id)

        word_id_to_tokens[word_id] = new_tokens

    changed_counts = {}
    for pair, delta in pair_deltas.items():
        new_count = pair_counts.get(pair, 0) + delta
        if new_count <= 0:
            pair_counts.pop(pair, None)
            changed_counts[pair] = 0
        else:
            pair_counts[pair] = new_count
            changed_counts[pair] = new_count

    return changed_counts

def split_on_special_tokens(text: str, special_tokens: list[str], keep_special_tokens = False):
    if not special_tokens:
        return [text]

    special_tokens = sorted(special_tokens, key=len, reverse=True)

    pattern = "|".join(re.escape(token) for token in special_tokens)
    if keep_special_tokens:
        return re.split(f"({pattern})", text)
    return re.split(pattern, text)  

def iter_pretokens(text: str, special_tokens=None, keep_special_tokens = False):
    special_tokens = special_tokens or []

    for chunk in split_on_special_tokens(text, special_tokens, keep_special_tokens):
        if chunk == "":
            continue
        if keep_special_tokens and chunk in special_tokens:
            yield chunk
            continue
        for match in re.finditer(PAT, chunk):
            yield match.group(0)

def merge_one_token_sequence(tokens, pair):
    merged_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            merged_tokens.append(pair[0] + pair[1])
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1
    return tuple(merged_tokens)

def count_chunk(args):
    input_path, start, end, special_tokens = args

    counts = defaultdict(int)

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")

    BYTE_TOKENS = tuple(bytes([i]) for i in range(256))

    for word in iter_pretokens(chunk, special_tokens):
        tokens = tuple(BYTE_TOKENS[b] for b in word.encode('utf-8'))
        counts[tokens] += 1
    return dict(counts)

def merge_counts(total_counts, chunk_counts):
    for tokens, count in chunk_counts.items():
        total_counts[tokens] += count

def build_token_counts_parallel(input_path, special_tokens, desired_num_chunks=64, num_processes=4):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, 
            desired_num_chunks=desired_num_chunks, 
            split_special_token=special_tokens[0].encode("utf-8") if special_tokens else b"<|endoftext|>",
        )

    args_list = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    total_counts = defaultdict(int)

    with Pool(processes=num_processes) as pool:
        for chunk_counts in pool.imap_unordered(count_chunk, args_list):
            merge_counts(total_counts, chunk_counts)

    return total_counts

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], profile = False):
    build_word_tokens_time = 0.0
    get_pairs_time = 0.0
    merge_time = 0.0
    
    word_id_to_tokens = {}
    word_id_to_count = defaultdict(int)

    if profile:
        print("profile:")

    start_time = time.perf_counter()
    token_counts = build_token_counts_parallel(
        input_path=input_path,
        special_tokens=special_tokens,
        desired_num_chunks=16,
        num_processes=4
    )
    
    for word_id, (tokens, count) in enumerate(token_counts.items()):
        word_id_to_tokens[word_id] = tokens
        word_id_to_count[word_id] = count

    build_word_tokens_time += time.perf_counter() - start_time

    if profile:
        print("  build_word_tokens_time:", build_word_tokens_time)

    pair_counts, pair_to_word_ids = build_pair_indexes(word_id_to_tokens, word_id_to_count)
    pair_heap = _PairMaxHeap(pair_counts)

    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256

    existing_tokens = set(vocab.values())
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in existing_tokens:
            vocab[next_id] = token_bytes
            existing_tokens.add(token_bytes)
            next_id += 1

    merges = []
    
    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        start_time = time.perf_counter()
        most_frequent_pair = pair_heap.pop_max(pair_counts)
        get_pairs_time += time.perf_counter() - start_time

        
        merges.append(most_frequent_pair)

        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[next_id] = new_token
        next_id += 1
        
        start_time = time.perf_counter()
        changed_counts = apply_merge_incremental(
            most_frequent_pair, word_id_to_tokens, word_id_to_count, pair_counts, pair_to_word_ids
        )
        for pair, count in changed_counts.items():
            pair_heap.update(pair, count)
        pair_heap.compact_if_needed(pair_counts)
        merge_time += time.perf_counter() - start_time

    
    if profile:    
        print("  get_pairs_time:", get_pairs_time)
        print("  merge_time:", merge_time)

    return vocab, merges

if __name__ == "__main__":
    vocab, merges = train_bpe(
        "toy.txt",
        vocab_size=266,
        special_tokens=["<|endoftext|>"]
    )

    print("vocab size:", len(vocab))
    print("last vocab items:")
    for i in range(256, len(vocab)):
        print(i, vocab[i])

    print("merges:", merges)