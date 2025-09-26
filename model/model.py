from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

#=============MODEL CONFIGURATION=============#
def model_config():
    MODEL_NAME = "zhihan1996/DNA_bert_6"
    MAX_LENGTH = 64
    BATCH_SIZE = 32
    EPOCHS = 4 # Fine-tuning requires fewer epochs
    LEARNING_RATE = 2e-5
    NUM_LABELS = 2

    return MODEL_NAME, MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, NUM_LABELS

#=============LOAD TOKENIZER & PRE-TRAINED MODEL=============#
def get_model(model_name, num_labels):
    print(f"Loading pre-trained model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model