import re
import os
import math

class BasicTokenizer:

    @staticmethod
    def tokenize(text):
        text = text.strip().lower()
        tokens = []

        # Match special tokens first, then words and punctuation
        pattern = r'(<\|startoftext\|>|<\|endoftext\|>)|(\w+)|([^\w\s])'
        for match in re.finditer(pattern, text):
            token = match.group(0)
            tokens.append(token)

        tokens = [token for token in tokens if token.split()]  # Remove empty tokens
        return tokens
    
    def build_vocab(self, texts):
        texts = ["<|startoftext|> " + text for text in texts]
        texts = " <|endoftext|> ".join(texts)
        tokens = self.tokenize(texts)
        tokens = sorted(set(tokens))
        self.reverse_vocabulary = {idx: token for idx, token in enumerate(tokens)}
        self.vocabulary = {token: idx for idx, token in self.reverse_vocabulary.items()}
        self.reverse_vocabulary[len(self.reverse_vocabulary)] = '<unk>'
        self.vocabulary['<unk>'] = len(self.vocabulary)

    def encode(self, text):
        text = "<|startoftext|> " + text + " <|endoftext|>"
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, self.vocabulary['<unk>']) for token in tokens]

    def decode(self, tokens):
        reverse_vocab = {idx: token for token, idx in self.vocabulary.items()}
        return ' '.join(reverse_vocab.get(token, '<unk>') for token in tokens)



# Tests
tokenizer = BasicTokenizer()
texts = []
data_dir = os.path.abspath('../data')

path, dirs, files = next(os.walk(data_dir))
# print("Files found:", files)
for file in files:
    if not file.endswith('.txt'):
        continue
    with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
        texts.append(f.read())

tokenizer.build_vocab(texts)

encodings = tokenizer.encode("This is; a! (test) sentence | heart <3 > : .")
print("Encoded:", encodings)
decoding = tokenizer.decode(encodings)
print("Decoded:", decoding)