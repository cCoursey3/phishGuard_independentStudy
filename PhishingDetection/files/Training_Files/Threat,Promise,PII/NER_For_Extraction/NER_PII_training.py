import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
import torch
import evaluate
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Load the data
input_file_path = r"C:\Users\Chloe\git\IndependentStudy\phishingDetection\PhishingDetection\files\Training_Files\Threat,Promise,PII\NER_For_Extraction\datasets\ner_processed.json"
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create a Hugging Face Dataset
dataset = Dataset.from_list(data)
# Split the dataset into train and validation sets
train_test_split = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})

# Load the tokenizer and model
model_name = "distilbert-base-uncased"  # Pretrained NER model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=4)  # Updated num_labels to 4

# Define the DataCollator
data_collator = DataCollatorForTokenClassification(tokenizer)

label_list = ["O", "pii_request", "threat", "promise"]
label_to_id = {label: i for i, label in enumerate(label_list)}

# Function to align labels with tokens
def align_labels_with_tokens(examples):
    tokenized_inputs = tokenizer(examples["row_text"], truncation=True, padding='max_length', max_length=512, return_offsets_mapping=True)
    labels = []

    for i in range(len(examples["labels"])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        original_labels = examples["labels"][i]

        # Align labels
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(original_labels[word_id])

        # Ensure new_labels has the same length as tokens with padding handled
        while len(aligned_labels) < 512:
            aligned_labels.append(-100)
        aligned_labels = aligned_labels[:512]  # Ensure it's not longer than max_length

        labels.append(aligned_labels)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply the function to the dataset
dataset = dataset.map(align_labels_with_tokens, batched=True)

# Check class distribution
label_counts = np.bincount([label for labels in dataset['train']['labels'] for label in labels if label != -100])
print("Class distribution in training set:", label_counts)

# Define class weights based on class distribution
total_labels = sum(label_counts)
class_weights = torch.tensor([total_labels / count for count in label_counts], dtype=torch.float)
class_weights = class_weights.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Custom Trainer class with custom loss function
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Compute metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [label for label in labels.flatten() if label != -100]
    true_predictions = [pred for pred, label in zip(predictions.flatten(), labels.flatten()) if label != -100]

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average='macro', zero_division=1)
    accuracy = accuracy_score(true_labels, true_predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Custom Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Save the model and tokenizer
output_dir = r"C:\Users\Chloe\git\IndependentStudy\phishingDetection\PhishingDetection\trained_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model and tokenizer saved to {output_dir}")
