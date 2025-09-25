# Genomic Sequence Classifier with DNABERT

Ever wondered if you can teach a computer to read and understand DNA? This project does just that. It uses a state-of-the-art Natural Language Processing model, **DNABERT**, to classify human DNA sequences and identify functional "promoter" regions, which are critical for gene activation.

This isn't just about getting a prediction; it's about using the model's internal "attention" mechanism to understand *why* it made its decision, highlighting the key DNA motifs that are biologically significant.

## \#\# Key Features

  * **Promoter Classification:** A binary classifier that distinguishes between promoter and non-promoter DNA sequences.
  * **State-of-the-Art Model:** Leverages a pre-trained **DNABERT** model, which understands the "language" of genomics.
  * **Fine-Tuning:** Built with PyTorch and the Hugging Face `transformers` library for efficient fine-tuning.
  * **Interpretable AI:** Includes the ability to visualize model attention, turning the "black box" into a tool for scientific insight.

-----

## \#\# How It Works

The model treats DNA as a language. A raw DNA sequence like `GATTACA...` is broken down into overlapping "words" of 6 letters (6-mers) by a specialized tokenizer. The pre-trained DNABERT model, which has already learned the fundamental grammar of DNA from massive genomic databases, is then fine-tuned on our specific task of promoter identification. This approach is highly effective and requires significantly less training time than starting from scratch.

-----

## \#\# Getting Started

Follow these steps to get the project up and running on your local machine.

### \#\#\# Prerequisites

  * Python 3.8+
  * PyTorch
  * A CUDA-enabled GPU is highly recommended for training.

### \#\#\# Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/harshrajhrj/genomic-sequence-classification-with-an-nlp-transformer.git
    cd genomic-sequence-classification-with-an-nlp-transformer
    ```

2.  **Download the dataset:**
    This project uses the "Human Gene Promoter and Non-Promoter Sequences" dataset.

      * Download it from [Kaggle](https://www.kaggle.com/datasets/zakarii/promoter-nonpromotor-dna-sequences).
      * Unzip the file and place `promoters.csv` in the root directory of this project.

3.  **Install dependencies:**
    It's recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```

-----

## \#\# Usage

The main script for training the model is `fine_tune_train.py`.

### \#\#\# Training the Model

To start fine-tuning the DNABERT model on the promoter dataset, run the following command in your terminal:

```bash
python fine_tune_train.py
```

The script will handle data preprocessing, training, and evaluation. It will print the progress for each epoch and save the best performing model weights as `dnabert_promoter_best_model.bin`.

### \#\#\# Making Predictions with the Trained Model

Once training is complete, you can easily use the saved model to make predictions on new DNA sequences. Here's a quick example snippet:

```python
# (You'll need to have the model, tokenizer, and label_encoder loaded)

# Your new DNA sequence
new_sequence = "cgcgcccgcgccgcatatacgcgtatatacgcgtatacgcgtatacgcgtacgcgta"

# Use a prediction function to get the result
# (See the implementation in `test_model.py` for a full example)
label, confidence = predict_sequence(new_sequence, model, tokenizer)

print(f"Sequence: {new_sequence[:30]}...")
print(f"Predicted Label: {label}")
print(f"Confidence: {confidence:.4f}")
```

-----

## \#\# Future Work

This project is a great foundation. Here are a few ideas for extending it:

  <!-- * [ ] Create a script for visualizing the attention heatmaps on promoter sequences. -->
  * [ ] Build a simple web interface with Streamlit or Flask to make predictions interactively.
  * [ ] Experiment with other pre-trained genomic models.
  * [ ] Expand the classifier to handle multi-class problems (e.g., classifying different types of genomic elements).

-----