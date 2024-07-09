## Import dependencies
import pandas as pd
import numpy as np
import os
from datasets import Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, EarlyStoppingCallback, GenerationConfig

## Load metrics
bleu = evaluate.load("bleu")
chrf = evaluate.load("chrf")
ter = evaluate.load("ter")
cter = evaluate.load("character")

## Define model name and main training parameters
model_dir = "gsarti"
model_name = "it5-base"
batch_size = 64
learning_rate = 5e-4
epochs = 10
finetuned_model_name = f"{model_name}_finetuned"

## Load dataset
# Create dataframes from .tsv files
df_train = pd.read_csv("data/train.csv", sep="\t").drop("SOURCE", axis=1)
df_val = pd.read_csv("data/val.csv", sep="\t").drop("SOURCE", axis=1)
df_test = pd.read_csv("data/test.csv", sep="\t")

# Turn into HuggingFace datasets
train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_val)
test_dataset = Dataset.from_pandas(df_test)

print(train_dataset, val_dataset, test_dataset)

## Tokenize
# Instantiate tokenizer
tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/{model_name}")

# Define tokenize function
def tokenize_function(references, targets):
    tokenized_data = tokenizer(
        text=references, # references
        text_target=targets, # labels
        # Padding and truncation are handled later by data collator
        padding=False,
        truncation=False
    )
    return tokenized_data

# Tokenize dataset
train_tokenized = train_dataset.map(
    tokenize_function,
    input_columns=["REFERENCE", "SCHWA"],
    batched=True,
    batch_size=batch_size,
    remove_columns=train_dataset.features
)

val_tokenized = val_dataset.map(
    tokenize_function,
    input_columns=["REFERENCE", "SCHWA"],
    batched=True,
    batch_size=batch_size,
    remove_columns=val_dataset.features
)

test_tokenized = test_dataset.map(
    tokenize_function,
    input_columns=["REF-G", "SCHWA"],
    batched=True,
    batch_size=batch_size,
    remove_columns=test_dataset.features
)

print(train_tokenized, val_tokenized, test_tokenized)

## Fine-tune
# Set max/min generation length for training and evaluation
train_generation_maxlen = len(max((train_tokenized["input_ids"] + val_tokenized["input_ids"]), key=len)) # Longest sequence in train/val set for training
eval_generation_maxlen = len(max(test_tokenized["input_ids"], key=len)) # Longest sequence in test set for evaluation
generation_minlen = len(min(test_tokenized["input_ids"], key=len)) # Shortest sequence in test set for both trainin and evaluation

# Instantiate model
it5 = AutoModelForSeq2SeqLM.from_pretrained(f"{model_dir}/{model_name}")

# Initialize data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=it5,
    padding=True, # pad to longest sequence in batch (no truncation)
    return_tensors="pt" # PyTorch tensors
)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=f"trainers/{finetuned_model_name}_trainer",
    overwrite_output_dir=True,
    evaluation_strategy="epoch", # Evaluate at the end of each epoch
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    num_train_epochs=epochs,
    logging_dir=f"tensorboard/{finetuned_model_name}",
    logging_strategy="epoch", # Log at the end of each epoch
    save_strategy="epoch", # Save checkpoints at the end of each epoch
    load_best_model_at_end=True,
    group_by_length=True, # Create batches based on sentences of similar length to minimize padding
    predict_with_generate=True,
    generation_max_length=train_generation_maxlen
)
# optimizer defaults: AdamW, no warmup, linear schedule
# evaluation defaults: uses eval_loss to determine best model + early stopping

# Initialize trainer
trainer = Seq2SeqTrainer(
    model = it5,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train
trainer.train()

# Export model
trainer.save_model(f"models/{finetuned_model_name}")
print("Model saved! Starting evaluation...")

## Evaluate
# Load finetuned tokenizer and model
new_tokenizer = AutoTokenizer.from_pretrained(f"models/{finetuned_model_name}")
new_model = AutoModelForSeq2SeqLM.from_pretrained(f"models/{finetuned_model_name}")

# Create directory to store results
os.makedirs(f"results/{finetuned_model_name}", exist_ok=True)

# Define function to compute and export metrics during evaluation
def compute_metrics(eval_pred):
    # Get predictions and labels
    predictions, labels = eval_pred

    # Replace -100 in labels with tokenizer's pad token ID (0) to decode them
    clean_labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(clean_labels, skip_special_tokens=True)
    decoded_labels_list = [[label] for label in decoded_labels]

    # Compute metrics
    bleu_scores = bleu.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        max_order=4 # Maximum n-gram dimension
    )

    chrf_score = chrf.compute(
        predictions=decoded_preds,
        references=decoded_labels_list
    )

    ter_score = ter.compute(
        predictions=decoded_preds,
        references=decoded_labels_list
    )

    character_scores = cter.compute(
        predictions=decoded_preds,
        references=decoded_labels_list,
        aggregate="mean", # Aggregate score is mean of scores for individual sentences
        return_all_scores=True # Return scores for individual sentences
    )

    # Get length of individual predictions
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]

    # Collect metrics
    metrics = {
        "BLEU": round(bleu_scores["bleu"], 4),
        "BLEU-2": round(bleu_scores["precisions"][1], 4),
        "BLEU-3": round(bleu_scores["precisions"][2], 4),
        "chrF": round(chrf_score["score"]/100, 4),
        "TER": round(ter_score["score"]/100, 4),
        "characTER_mean": round(character_scores["cer_score"], 4),
        "mean_gen_len": np.mean(prediction_lens)
    }
    
    # Export predictions along with labels and sentence-level characTER score
    with open(f"results/{finetuned_model_name}/predictions.csv", "a+", encoding="utf8") as wf:
        wf.write("OUTPUT" + "\t" + "TARGET" + "\t" + "characTER_SCORE" + "\n")
        for i in range(len(decoded_preds)):
            wf.write(decoded_preds[i] + "\t" + decoded_labels[i] + "\t" + str(character_scores["cer_scores"][i]) + "\n")
    
    # Export other metrics separately
    with open(f"results/{finetuned_model_name}/metrics.csv", "a+", encoding="utf8") as wf:
        wf.write("\t".join(key for key in metrics.keys()) + "\n")
        wf.write("\t".join(str(value) for value in metrics.values()) + "\n")

    return metrics

# Define generation configuration
generation_config = GenerationConfig(
        # Special token IDs
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
        # Min/max lengths for generation
        max_new_tokens=eval_generation_maxlen,
        min_new_tokens=generation_minlen,
        # Beam search
        num_beams=5,
        # Multinomial sampling
        do_sample=True
)

# Define new arguments and trainer for evaluation
test_args = Seq2SeqTrainingArguments(
    output_dir=f"results/{finetuned_model_name}",
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    group_by_length=True,
    generation_config=generation_config
)

test_trainer = Seq2SeqTrainer(
    model=new_model,
    args=test_args,
    data_collator=data_collator,
    eval_dataset=test_tokenized,
    tokenizer=new_tokenizer,
    compute_metrics=compute_metrics
)

# Run evaluation
print("Running evaluation...")
test_trainer.evaluate()
print(f"Evaluation done! Results saved in results/{finetuned_model_name}")