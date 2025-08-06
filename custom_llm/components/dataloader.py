from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):

    def __init__(self, text, tokenizer, context_length=32, stride=16):

        super().__init__()

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride
        self.input_ids, self.target_ids = self._create_dataset(text)

    
    def _create_dataset(self, text):

        x = []
        y = []

        tokens = self.tokenizer.encode(text)

        print(f"Total tokens: {len(tokens)}")

        for i in range(0, len(tokens) - self.context_length, self.stride):
            window_data = tokens[i:i+self.context_length]
            window_target = tokens[i+1:i+self.context_length+1]

            x.append(window_data)
            y.append(window_target)

        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_data_loader(batch_size, text, tokenizer, shuffle=True, num_workers=0, drop_last=True, context_length=32, stride=16):
    
    dataset = CustomDataset(text, tokenizer, context_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)