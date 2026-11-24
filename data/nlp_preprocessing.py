import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        combined=positive + negative
        words = sorted(set(" ".join(combined).split()))
        stoi = {word:i for i,word in enumerate(words, start=1)}
        encode = lambda s: [stoi[w] for w in s.split()]
        # 2. Encode each sentence by replacing words with their IDs
        combined_tensors = []
        for s in combined:
            encoded = encode(s)
            tensor = torch.tensor(encoded)
            combined_tensors.append(tensor)
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        return nn.utils.rnn.pad_sequence(combined_tensors, batch_first=True)
