from collections import defaultdict

from cs336_basics.bpe import _PairMaxHeap, apply_merge_incremental, build_pair_indexes, iter_pairs


def recompute_indexes(word_id_to_tokens, word_id_to_count):
    pair_counts = defaultdict(int)
    pair_to_word_ids = defaultdict(set)
    for word_id, tokens in word_id_to_tokens.items():
        for pair in iter_pairs(tokens):
            pair_counts[pair] += word_id_to_count[word_id]
            pair_to_word_ids[pair].add(word_id)
    return dict(pair_counts), dict(pair_to_word_ids)


def test_pair_max_heap_preserves_count_then_lexicographic_order():
    counts = {
        (b"a", b"z"): 4,
        (b"b", b"a"): 4,
        (b"z", b"a"): 3,
    }
    heap = _PairMaxHeap(counts)
    assert heap.pop_max(counts) == max(counts, key=lambda pair: (counts[pair], pair))


def test_pair_max_heap_discards_stale_entries_and_compacts():
    counts = {(bytes([i]), b"x"): i + 1 for i in range(100)}
    heap = _PairMaxHeap(counts)
    changed = (b"\x00", b"x")
    counts[changed] = 1000
    heap.update(changed, counts[changed])
    for _ in range(500):
        heap.update(changed, counts[changed])
    assert heap.pop_max(counts) == changed
    heap.compact_if_needed(counts)
    assert len(heap.heap) <= max(64, 2 * len(counts))


def test_incremental_pair_deltas_match_full_recomputation():
    word_id_to_tokens = {
        0: (b"a", b"b", b"a", b"b"),
        1: (b"a", b"b", b"c"),
        2: (b"c", b"a", b"b"),
        3: (b"q",),
    }
    word_id_to_count = {0: 2, 1: 3, 2: 1, 3: 9}
    pair_counts, pair_to_word_ids = build_pair_indexes(word_id_to_tokens, word_id_to_count)

    for merge in ((b"a", b"b"), (b"ab", b"ab"), (b"c", b"ab")):
        apply_merge_incremental(
            merge,
            word_id_to_tokens,
            word_id_to_count,
            pair_counts,
            pair_to_word_ids,
        )
        expected_counts, expected_word_ids = recompute_indexes(word_id_to_tokens, word_id_to_count)
        assert pair_counts == expected_counts
        assert dict(pair_to_word_ids) == expected_word_ids


def test_heap_and_incremental_updates_match_full_recomputation_over_many_merges():
    import random

    rng = random.Random(20261013)
    alphabet = [bytes([97 + i]) for i in range(5)]
    word_id_to_tokens = {
        word_id: tuple(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
        for word_id in range(80)
    }
    word_id_to_count = {
        word_id: rng.randint(1, 20)
        for word_id in word_id_to_tokens
    }
    pair_counts, pair_to_word_ids = build_pair_indexes(word_id_to_tokens, word_id_to_count)
    heap = _PairMaxHeap(pair_counts)

    for _ in range(120):
        if not pair_counts:
            break
        expected = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        best_pair = heap.pop_max(pair_counts)
        assert best_pair == expected
        changed_counts = apply_merge_incremental(
            best_pair,
            word_id_to_tokens,
            word_id_to_count,
            pair_counts,
            pair_to_word_ids,
        )
        for pair, count in changed_counts.items():
            heap.update(pair, count)
        heap.compact_if_needed(pair_counts)

        expected_counts, expected_word_ids = recompute_indexes(word_id_to_tokens, word_id_to_count)
        assert pair_counts == expected_counts
        assert dict(pair_to_word_ids) == expected_word_ids
