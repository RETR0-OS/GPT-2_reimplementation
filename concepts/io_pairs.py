import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):

    def __init__(self, txt, tokenizer, max_len = 32, stride=1):
        self.tokenizer = tokenizer
        self.data = txt
        self.context_window = max_len
        self.index = 0
        self.stride = stride
        self.input_ids, self.target_ids = self._create_dataset(txt)
        self.idx = 0

    def _create_dataset(self, txt):
        x = []
        y = []

        enocoded_data = self.tokenizer.encode(txt)

        try:
            for i in range(0, len(enocoded_data) - self.context_window, self.stride):
                window_data = enocoded_data[i:i + self.context_window]
                window_target = enocoded_data[i+1: i + self.context_window + 1]
                x.append(window_data)
                y.append(window_target)
        except IndexError:
            pass
        finally:
            if len(enocoded_data) % self.context_window != 0:
                x.append(enocoded_data[-self.context_window:])
                y.append(enocoded_data[-self.context_window+1:]) 

            # Pad all sequences in x and y to context_window length
            import torch.nn.functional as F
            x_padded = [F.pad(torch.tensor(seq, dtype=torch.long), (0, self.context_window - len(seq)), value=0) for seq in x]
            y_padded = [F.pad(torch.tensor(seq if isinstance(seq, list) else [seq], dtype=torch.long), (0, self.context_window - len(seq if isinstance(seq, list) else [seq])), value=0) for seq in y]
            return torch.stack(x_padded), torch.stack(y_padded)
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt, batch_size=32, max_len=32, stride=32, shuffle=True, drop_last=True, num_workers=0):
    dataset = CustomDataset(txt, tiktoken.get_encoding("gpt2"), max_len, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


if __name__ == "__main__":

    f = open("C:/Users/aadit/Projects/LLM_from_scratch/data/the_verdict.txt", "r")
    dataloader = create_dataloader(f.read(), batch_size=1, max_len=256, stride=128, shuffle=False, drop_last=True, num_workers=0)
    print("Dataset length:", len(dataloader))

    iterator = iter(dataloader)
    batch = next(iterator)
    print("First batch input IDs:", batch[0])
    print("First batch target IDs:", batch[1])
    batch = next(iterator)
    print("Second batch input:", batch[0])
    print("Second batch input:", batch[1])

    


        
