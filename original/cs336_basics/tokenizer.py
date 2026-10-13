from cs336_basics.bpe import iter_pretokens
import pickle

def merge_token_sequence(tokens, pair_to_merge_index):
    tokens = list(tokens)
    while True:
        best_pair = None
        best_index = float("inf")
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in pair_to_merge_index and pair_to_merge_index[pair] < best_index:
                best_pair = pair
                best_index = pair_to_merge_index[pair]

        if best_pair is None:
            break

        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                new_tokens.append(best_pair[0] + best_pair[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    return tokens

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.token_to_id = {token: token_id for token_id, token in vocab.items()}
        self.id_to_token = {token_id: token for token_id, token in vocab.items()}

        self.pair_to_merge_index = {pair: i for i, pair in enumerate(merges)}

        for special_token in self.special_tokens:
            special_token = special_token.encode("utf-8")
            if special_token not in self.token_to_id:
                word_id = max(self.vocab.keys()) + 1
                self.vocab[word_id] = special_token
                self.token_to_id[special_token] = word_id
                self.id_to_token[word_id] = special_token

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_path, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str):
        ids = []
        for word in iter_pretokens(text, self.special_tokens, keep_special_tokens=True):
            if word in self.special_tokens:
                ids.append(self.token_to_id[word.encode("utf-8")])
                continue
            tokens = tuple(bytes([b]) for b in word.encode("utf-8"))
            merged_tokens = merge_token_sequence(tokens, self.pair_to_merge_index)
            ids.extend(self.token_to_id[token] for token in merged_tokens)
        return ids

    def encode_iterable(self, iterable):
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    
    def decode(self, ids):
        text = b"".join(self.id_to_token[id] for id in ids).decode("utf-8", errors="replace")
        return text