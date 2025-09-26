import torch
from torch.utils.data import Dataset

#=============CREATE PYTORCH DATASET=============#
class DNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        label = self.labels[idx]

        # Tokenize the sequence
        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Squeeze to remove the batch dimension
        # item = {key: val.squeeze(0) for key, val in encoding.items()}
        # item['labels'] = torch.tensor(label, dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
# class SequenceDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, index):
#         item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[index])
#         return item
    
#     def __len__(self):
#         return len(self.labels)


# print(df.head())