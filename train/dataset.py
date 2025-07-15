import torch
from torch.utils.data import Dataset

class MultimodalFinancialDataset(Dataset):
    def __init__(self, data):
        self.tabular = data['tabular']
        self.text = data['text']
        self.images = data['images']
        self.metadata = data['metadata']
        self.labels = data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'tabular': torch.tensor(self.tabular[idx], dtype=torch.float32),
            'text': torch.tensor(self.text[idx], dtype=torch.float32),
            'image': torch.tensor(self.images[idx], dtype=torch.float32),
            'metadata': torch.tensor(self.metadata[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    # def __getitem__(self, index):
    #     return {
    #         'tabular': torch.tensor(self.data['tabular'][index], dtype=torch.float32),
    #         'text': torch.tensor(self.data['text'][index], dtype=torch.float32),
    #         'images': torch.tensor(self.data['images'][index], dtype=torch.float32),
    #         'metadata': torch.tensor(self.data['metadata'][index], dtype=torch.float32),
    #         'labels': torch.tensor(self.data['labels'][index], dtype=torch.long)  # Ensure labels are long integers
    # }