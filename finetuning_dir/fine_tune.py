import argparse
import sys
import logging
import os
from datasets import Dataset
import torch
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# To avoid the INVALID_PARAMETER_VALUE error in MLflow, disable MLflow integration
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"

logger = logging.getLogger(__name__)

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING
)

# Model and tokenizer setup
pretrained_model_name = "microsoft/Phi-3-mini-4k-instruct"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map=None,
    attn_implementation="eager"
)

def initialize_model_and_tokenizer(pretrained_model_name, model_kwargs):
    """
    Initialize the model and tokenizer with the given pretrained model name and arguments.
    """
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
    return model, tokenizer

def preprocess_function(examples, tokenizer):
    """
    Preprocess function for tokenizing the dataset.
    """
    tokens = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

def load_and_preprocess_data(train_filepath, test_filepath, tokenizer):
    """
    Load and preprocess the dataset.
    """
    train_dataset = Dataset.from_json(train_filepath)
    test_dataset = Dataset.from_json(test_filepath)
    train_dataset = train_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda examples: preprocess_function(examples, tokenizer), batched=True)
    return train_dataset, test_dataset

def main(train_file, eval_file, model_output_dir):
    """
    Main function to fine-tune the model.
    """
    with mlflow.start_run():
        model, tokenizer = initialize_model_and_tokenizer(pretrained_model_name, model_kwargs)
        train_dataset, test_dataset = load_and_preprocess_data(train_file, eval_file, tokenizer)

        # Fine-tuning settings
        finetuning_settings = {
            "bf16": True,
            "do_eval": True,
            "output_dir": model_output_dir,
            "eval_strategy": "epoch",
            "learning_rate": 1e-4,
            "logging_steps": 20,
            "lr_scheduler_type": "linear",
            "num_train_epochs": 3,
            "overwrite_output_dir": True,
            "per_device_eval_batch_size": 4,
            "per_device_train_batch_size": 4,
            "remove_unused_columns": True,
            "save_steps": 500,
            "seed": 0,
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.2,
        }

        training_args = TrainingArguments(
            **finetuning_settings
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            max_seq_length=2048,
            dataset_text_field="text",
            tokenizer=tokenizer,
            packing=True
        )

        train_result = trainer.train()

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)

        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            artifact_path=model_output_dir,  # This is a relative path to save model files within MLflow run
        )

        # Evaluation
        tokenizer.padding_side = 'left'
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(test_dataset)
        trainer.log_metrics("eval", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, required=True, help="Path to the training data")
    parser.add_argument("--eval-file", type=str, required=True, help="Path to the evaluation data")
    parser.add_argument("--model_output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    args = parser.parse_args()
    main(args.train_file, args.eval_file, args.model_output_dir)
