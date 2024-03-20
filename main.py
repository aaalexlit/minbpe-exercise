import os
from collections import Counter


class BasicTokenizer:

    def __init__(self):
        self.merges = {}
        self.vocab = self._precalcluate_vocab()

        
    def train(self, text, vocab_size, verbose=False):
        tokens = text.encode("utf-8") # raw bytes
        tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
        num_merges = vocab_size - 256 # number of merges to achieve desired vocab size
        ids = list(tokens) # copy so we don't destroy the original list
        for i in range(num_merges):
            stats = self.get_stats(ids)
            top_pair = max(stats, key=stats.get)
            new_idx = 256 + i
            ids = self.merge(ids, top_pair, new_idx)
            self.merges[top_pair] = new_idx
            if verbose:
                print(f'merged {top_pair} into a new token {new_idx}')
        self._precalcluate_vocab()


    def encode(self, text):
        tokens = list(text.encode("utf-8")) # raw bytes
        for pair, token in self.merges.items():
            tokens = self.merge(ids=tokens, pair=pair, idx=token)
        return tokens


    def decode(self, ids):
        tokens = b''.join(self.vocab[idx] for idx in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text


    def _precalcluate_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab


    @staticmethod
    def get_stats(ids):
        return Counter([pair for pair in zip(ids, ids[1:])])


    @staticmethod
    def merge(ids, pair, idx):
        i = 0
        res = []

        while i < len(ids):
            # make sure to append the last token and not to get out of range
            if i == len(ids) - 1:
                res.append(ids[i])
                i += 1
            elif (ids[i], ids[i+1]) == pair:
                res.append(idx)
                i += 2
            else:
                res.append(ids[i])
                i += 1

        return res


if __name__ == '__main__':
    dirname = os.path.dirname(os.path.abspath(__file__))
    taylorswift_file = os.path.join(dirname, 'taylorswift.txt')
    contents = open(taylorswift_file, "r", encoding="utf-8").read()
    basic_tokenizer = BasicTokenizer()
    basic_tokenizer.train(contents, 1000)
    assert contents == basic_tokenizer.decode(basic_tokenizer.encode(contents))
    print('merges')
    print(basic_tokenizer.merges)
    
