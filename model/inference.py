import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from model import model_config, get_model

model_config = model_config()
MODEL_NAME = model_config[0]
MAX_LENGTH = model_config[1]
NUM_LABELS = model_config[5]

class_labels = np.array(['Non-Promoter', 'Promoter'])

# Load the specialized tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the model architecture
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

# Load the fine-tuned weights you saved
model.load_state_dict(torch.load('dnabert_promoter_best_model.bin'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Model loaded and ready for inference!")


def predict_sequence(sequence):
    """
    Takes a raw DNA sequence and returns the predicted label and confidence score.
    """
    # Tokenize the sequence
    inputs = tokenizer(
        sequence,
        return_tensors="pt", # Return PyTorch tensors
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        # Get the raw model output (logits)
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_class_id = torch.max(probabilities, dim=1)

    predicted_label = class_labels[predicted_class_id.item()]

    return predicted_label, confidence.item()

# load test set
import pandas as pd
test_df = pd.read_csv('../dataset/test_set.csv')
print(f"Loaded {len(test_df)} sequences from the test set.")
# evaluate on the test set
results = []
for idx, row in test_df.iterrows():
    seq = row['sequence']
    true_label = row['label']
    pred_label, confidence = predict_sequence(seq)
    results.append({
        'sequence': seq,
        'true_label': true_label,
        'predicted_label': pred_label,
        'confidence': confidence
    })
results_df = pd.DataFrame(results)
accuracy = (results_df['true_label'] == results_df['predicted_label']).mean()
print(f"Test Set Accuracy: {accuracy:.4f}")

# save results to csv
results_df.to_csv('../dataset/test_set_predictions.csv', index=False)

# # Example of a real human promoter sequence (TATA-box)
# new_dna_sequence = "cgcgcccgcgccgcatatacgcgtatatacgcgtatacgcgtatacgcgtacgcgta"

# # Get the prediction
# label, confidence = predict_sequence(new_dna_sequence)

# print(f"\nSequence:   {new_dna_sequence[:40]}...")
# print(f"Prediction: {label}")
# print(f"Confidence: {confidence:.4f}")

# # Example of a random, non-promoter-like sequence
# random_sequence = "atcgatcgatcgatcgatcgatcgatcgatcgatcgatcgatcgatcgatcgatc"
# label, confidence = predict_sequence(random_sequence)
# print(f"\nSequence:   {random_sequence[:40]}...")
# print(f"Prediction: {label}")
# print(f"Confidence: {confidence:.4f}")