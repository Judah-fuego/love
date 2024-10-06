from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the dataset
ds = load_dataset("ieuniversity/flirty_or_not")

# Set up model and tokenizer
model_name = "bert-base-uncased"  # Pretrained model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification (flirty or not)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocess the dataset by tokenizing text
def preprocess_function(examples):
    return tokenizer(examples['texts'], truncation=True, padding='max_length', max_length=128)

# Tokenize the dataset
tokenized_datasets = ds.map(preprocess_function, batched=True)

# Split the dataset into training and evaluation sets
train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['test']

# Metrics for evaluation (accuracy, precision, recall, F1)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Define training arguments
# Define training arguments
training_args = TrainingArguments(
    output_dir='./results3',                  # output directory
    num_train_epochs=6,                      # number of training epochs
    per_device_train_batch_size=16,          # batch size for training
    per_device_eval_batch_size=16,           # batch size for evaluation
    evaluation_strategy="epoch",             # evaluate each epoch
    save_strategy="epoch",                   # save the model each epoch
    logging_dir='./logs',                    # directory for logging
    logging_steps=5000,                      # log every 5000 steps (reduce frequency)
    log_level="error",                       # only log errors
    load_best_model_at_end=True,             # load best model when finished training
    disable_tqdm=True                        # disable progress bars (optional)
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics  # Use the compute_metrics function for evaluation
)

# Train the model
trainer.train()

# Evaluate the model on the test set
trainer.evaluate()

# Save the model and tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")


