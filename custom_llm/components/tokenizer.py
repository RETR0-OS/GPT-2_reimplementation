import tiktoken
from enum import Enum
from typing import List, Literal

class TokenizerMap(Enum):
    GPT_2 = tiktoken.get_encoding("gpt2")
    GPT_3 = tiktoken.get_encoding("cl100k_base")

class BytePairTokenizer:

    def __init__(self, tokenizer_type: Literal["GPT_2", "GPT_3"] ="GPT_2"):
        """
            Initialize a Byte-Pair Tokenizer for GPT-2 or GPT-3

            @param tokenizer_type: The type of tokenizer to initialize (one of 'GPT_2' or 'GPT_3')
        """

        if tokenizer_type == "GPT_2":
            self.tokenizer = TokenizerMap.GPT_2.value
        elif tokenizer_type == "GPT_3":
            self.tokenizer = TokenizerMap.GPT_3.value
        else:
            raise NotImplementedError("Sorry, this tokenizer is not supported. Please choose one of 'GPT_2' or 'GPT_3'")
        
    def encode(self, text: str):
        """
            Encode text to corresponding class IDs.

            @param text: text to encode
        """
        return self.tokenizer.encode(text)
    
    def decode(self, ids: List[int]):
        """
            Decode predicted class IDs to text

            @param ids: List of token IDs to decode.
        """
        return self.tokenizer.decode(ids)
    
    def __len__(self):
        """
            Get vocabulary size if the chosen tokenizer
        """
        return self.tokenizer.n_vocab

