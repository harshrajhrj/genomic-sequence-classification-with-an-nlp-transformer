import pandas as pd
import numpy as np
import random
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from tqdm.auto import tqdm

#=============Hugging Face Login (if needed)=============#
from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))

#=============Custom Imports=============
from model import model_config, get_model
from dataset import DNADataset

#=============SET SEED FOR REPRODUCIBILITY=============#
def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#=============LOAD AND PREPARE DATA=============#
def load_data(file_path):
    """Loads, preprocesses, and splits the dataset."""
    try:
        df = pd.read_csv(file_path)
    except:
        print("Dataset not found. Please ensure the dataset is in the correct path.")
        exit()

    df = df.rename(columns={'Label': 'label', 'Sequence': 'sequence'})

    #=============ENCODE LABELS=============#
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])

    #=============Split data=============#
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    #=============Split val=============#
    val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
    # save test set for final evaluation
    test_df.to_csv('../dataset/test_set.csv', index=False)

    return train_df, val_df, label_encoder

#=============Training & Evaluation Loop=============#
def train_epoch(model, data_loader, optimizer, device, scheduler):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    actual_labels = []
    # Use tqdm for a user-friendly progress bar
    for batch in tqdm(data_loader, desc="Training"):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        loss.backward()

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        # Calculate training accuracy
        _, preds = torch.max(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        actual_labels.extend(labels.cpu().numpy())
        
    train_accuracy = accuracy_score(actual_labels, predictions)
    return total_loss / len(data_loader), train_accuracy

def eval_model(model, data_loader, device):
    """Evaluates the model on the validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, report

def main(args):
    """Main function to orchestrate the fine-tuning process."""
    set_seed(args.seed)

    #=============LOAD AND PREPARE DATA=============#
    train_df, val_df, label_encoder = load_data(args.data_path)
    
    #=============LOAD TOKENIZER & PRE-TRAINED MODEL=============#
    # model_config = model_config()
    tokenizer, model = get_model(args.model_name, num_labels=len(label_encoder.classes_)) # len(label_encoder.classes_)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    model.to(device)

    #=============TOKENIZE DATA=============#
    print("Tokenizing data...")
    train_dataset = DNADataset(
        sequences=train_df['sequence'].tolist(),
        labels=train_df['label_encoded'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    val_dataset = DNADataset(
        sequences=val_df['sequence'].tolist(),
        labels=val_df['label_encoded'].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    #=============SETUP OPTIMIZER AND SCHEDULER=============#
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    #=============Main Training Process=============#
    best_accuracy = 0
    best_val_loss = float('inf')
    
    print("\nStarting fine-tuning...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scheduler)
        val_loss, val_acc, report = eval_model(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.bin")
            print("New best model saved!")
    
    print("\n--- Final Evaluation Report ---")
    print(report)
    print("\nFine-tuning complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DNABERT for promoter classification.")
    parser.add_argument("--data_path", type=str, default="../dataset/dataset.csv", help="Path to the dataset CSV file.")
    parser.add_argument("--model_name", type=str, default="zhihan1996/DNA_bert_6", help="Name of the pre-trained model from Hugging Face.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the best model.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training and validation batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer.")
    parser.add_argument("--max_length", type=int, default=64, help="Max sequence length for tokenizer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    args = parser.parse_args()
    main(args)