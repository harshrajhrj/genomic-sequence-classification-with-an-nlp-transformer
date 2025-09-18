#====================IMPORT LIBRARIES====================#
from logging import config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tqdm.auto import tqdm

print("Libraries imported successfully.")
print("-"*50)

#====================CONFIG DATASET====================#
class Config_Dataset:
    def __init__(self, K_VALUE=6):
        self.df = pd.read_csv('../dataset/dataset.csv')
        self.K_VALUE = K_VALUE

    def info(self):
        print("Dataset Shape:", self.df.shape)
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nClass Distribution:")
        print(self.df['Label'].value_counts())

    def generate_kmers(self, sequence):
        """Generates a list of k-mers from a DNA sequence."""
        kmers = []
        num_kmers = len(sequence) - self.K_VALUE + 1
        for i in range(num_kmers):
            kmer = sequence[i:i + self.K_VALUE]
            kmers.append(kmer)
        return kmers
    
    def encode_labels(self):
        """Encode the labels into numerical format."""
        self.label_encoder = LabelEncoder()
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['Label'])
        # print("\nLabels and their encoded values:")
        # print(self.df[['Label', 'label_encoded']].value_counts())

    def split_data(self, split_size=0.3, val_size=0.5, random_state=42):
        """Split the dataset into training, validation, and test sets.
        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            val_size (float): Proportion of the test set to include in the validation split.
            random_state (int): Random seed for reproducibility.
        Returns:
            train_df (DataFrame): Training set.
            val_df (DataFrame): Validation set.
            test_df (DataFrame): Test set.
        """
        train_df, temp_df = train_test_split(
            self.df,
            test_size=split_size,  # 30% for temp (will be split into val and test)
            random_state=random_state,
            stratify=self.df['label_encoded']
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=val_size,  # Split the 30% into 15% val and 15% test
            random_state=random_state,
            stratify=temp_df['label_encoded']
        )

        # print(f"\nTraining set size: {len(train_df)}")
        # print(f"Validation set size: {len(val_df)}")
        # print(f"Test set size: {len(test_df)}")

        return train_df, val_df, test_df
    
    def build_vocab(self, train_data_frame):
        """Create a set of all unique k-mers from the training data."""
        all_kmers = set()
        for text in train_data_frame['kmers']:
            for kmer in text.split():
                all_kmers.add(kmer)

        # Build the vocabulary mapping
        # Reserve IDs for special tokens
        kmer_to_id = {kmer: i+4 for i, kmer in enumerate(sorted(list(all_kmers)))}
        kmer_to_id['[PAD]'] = 0  # Padding token
        kmer_to_id['[UNK]'] = 1  # Unknown token
        kmer_to_id['[CLS]'] = 2  # Classification token (start of sequence)
        kmer_to_id['[SEP]'] = 3  # Separator token (end of sequence)

        # Create the reverse mapping
        id_to_kmer = {id: kmer for kmer, id in kmer_to_id.items()}

        vocab_size = len(kmer_to_id)
        # print(f"Vocabulary Size: {vocab_size}")

        return kmer_to_id, id_to_kmer, vocab_size

    def preprocess(self):
        # Apply the function to our sequence column
        # We'll join the k-mers into a single string separated by spaces, like a sentence.
        self.df['kmers'] = self.df['Sequence'].apply(lambda seq: " ".join(self.generate_kmers(seq)))

        # print("\nExample after K-merization:")
        # print(self.df[['Sequence', 'kmers']].head())

        self.encode_labels()
        train_df, val_df, test_df = self.split_data(split_size=0.3, val_size=0.5, random_state=42)
        kmer_to_id, id_to_kmer, VOCAB_SIZE = self.build_vocab(train_df)

        return train_df, val_df, test_df, kmer_to_id, id_to_kmer, VOCAB_SIZE

#====================CUSTOM PYTORCH DATASET====================#
class PromoterDataset(Dataset):
    def __init__(self, texts, labels, kmer_to_id, max_length):
        self.texts = texts
        self.labels = labels
        self.kmer_to_id = kmer_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 1. Tokenize: Convert k-mers to integer IDs
        # Split sentence into k-mers and add CLS/SEP tokens
        kmer_list = text.split()
        token_ids = [self.kmer_to_id.get('[CLS]')]
        token_ids.extend([self.kmer_to_id.get(k, self.kmer_to_id['[UNK]']) for k in kmer_list])
        token_ids.append(self.kmer_to_id.get('[SEP]'))

        # 2. Pad or truncate the sequence
        if len(token_ids) < self.max_length:
            padding_len = self.max_length - len(token_ids)
            token_ids.extend([self.kmer_to_id['[PAD]']] * padding_len)
        else:
            token_ids = token_ids[:self.max_length-1] + [self.kmer_to_id.get('[SEP]')]

        # 3. Create attention mask
        attention_mask = [1 if id != self.kmer_to_id['[PAD]'] else 0 for id in token_ids]

        # 4. Convert to PyTorch tensors
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
#====================INSTANTIATE DATASET AND DATALOADERS====================#
class PrepareDataLoaders:
    def __init__(self, K_VALUE=6, BATCH_SIZE=32, MAX_LENGTH=64):
        self.K_VALUE = K_VALUE
        self.BATCH_SIZE = BATCH_SIZE
        self.MAX_LENGTH = MAX_LENGTH

    def create_dataset_instances(self, train_df, val_df, test_df, kmer_to_id):
        train_dataset = PromoterDataset(
            texts=train_df['kmers'].tolist(),
            labels=train_df['label_encoded'].tolist(),
            kmer_to_id=kmer_to_id,
            max_length=self.MAX_LENGTH
        )

        val_dataset = PromoterDataset(
            texts=val_df['kmers'].tolist(),
            labels=val_df['label_encoded'].tolist(),
            kmer_to_id=kmer_to_id,
            max_length=self.MAX_LENGTH
        )

        test_dataset = PromoterDataset(
            texts=test_df['kmers'].tolist(),
            labels=test_df['label_encoded'].tolist(),
            kmer_to_id=kmer_to_id,
            max_length=self.MAX_LENGTH
        )

        # print(f"Train Dataset Size: {len(train_dataset)}")
        # print(f"Validation Dataset Size: {len(val_dataset)}")
        # print(f"Test Dataset Size: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, train_df, val_df, test_df, kmer_to_id):
        train_dataset, val_dataset, test_dataset = self.create_dataset_instances(train_df, val_df, test_df, kmer_to_id)

        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.BATCH_SIZE, shuffle=False)

        return train_loader, val_loader, test_loader

    def initiate(self, train_df, val_df, test_df, kmer_to_id):
        train_loader, val_loader, test_loader = self.create_dataloaders(train_df, val_df, test_df, kmer_to_id)

        # print(f"Number of Batches in Train Loader: {len(train_loader)}")
        # print(f"Number of Batches in Validation Loader: {len(val_loader)}")
        # print(f"Number of Batches in Test Loader: {len(test_loader)}")

        return train_loader, val_loader, test_loader
    
#====================MAIN EXECUTION====================#
from transformers import BertConfig, BertForSequenceClassification

class GenomicClassifier(nn.Module):
    def __init__(self, VOCAB_SIZE, MAX_LENGTH):
        super(GenomicClassifier, self).__init__()

        # Define model configuration
        self.config = BertConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=256,                    # Dimension of the encoder layers
            num_hidden_layers=4,                # Number of hidden layers
            num_attention_heads=4,              # Number of attention heads
            intermediate_size=1024,             # Dimension of the "feed-forward" layer
            max_position_embeddings=MAX_LENGTH,
            num_labels=2,                       # Binary classification (Promoter/Non-Promoter)
            hidden_dropout_prob=0.3,            # Default is 0.1
            attention_probs_dropout_prob=0.3    # Default is 0.1
        )

        # Instantiate the model with the new configuration
        self.model = BertForSequenceClassification(self.config)

if __name__ == "__main__":
    # Set hyperparameters
    MAX_LENGTH = 64  # Max k-mers per sequence (DNA length is 57, so ~52 kmers + special tokens)
    BATCH_SIZE = 32
    config_dataset = Config_Dataset(K_VALUE=6)
    train_df, val_df, test_df, kmer_to_id, id_to_kmer, VOCAB_SIZE = config_dataset.preprocess()
    config_dataloaders = PrepareDataLoaders(K_VALUE=6, BATCH_SIZE=32, MAX_LENGTH=64)
    train_loader, val_loader, test_loader = config_dataloaders.initiate(train_df, val_df, test_df, kmer_to_id)

    model = GenomicClassifier(VOCAB_SIZE, MAX_LENGTH).model

    # Check the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel instantiated with {num_params:,} trainable parameters (BERT-base).")

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to device: {device}")