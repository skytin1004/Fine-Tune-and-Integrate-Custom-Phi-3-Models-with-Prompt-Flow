import argparse
import sys
import logging
import os
from datasets import load_dataset
import torch
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# To avoid the INVALID_PARAMETER_VALUE error in MLflow, disable MLflow integration
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "True"

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.WARNING
)
logger = logging.getLogger(__name__)

def initialize_model_and_tokenizer(model_name, model_kwargs):
    """
    Initialize the model and tokenizer with the given pretrained model name and arguments.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 2048
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
    return model, tokenizer

def apply_chat_template(example, tokenizer):
    """
    Apply a chat template to tokenize messages in the example.
    """
    messages = example["messages"]
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return example

def load_and_preprocess_data(train_filepath, test_filepath, tokenizer):
    """
    Load and preprocess the dataset.
    """
    train_dataset = load_dataset('json', data_files=train_filepath, split='train')
    test_dataset = load_dataset('json', data_files=test_filepath, split='train')
    column_names = list(train_dataset.features)

    train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to train dataset",
    )

    test_dataset = test_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to test dataset",
    )

    return train_dataset, test_dataset

def train_and_evaluate_model(train_dataset, test_dataset, model, tokenizer, output_dir):
    """
    Train and evaluate the model.
    """
    training_args = TrainingArguments(
        bf16=True,
        do_eval=True,
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=5.0e-06,
        logging_steps=20,
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        overwrite_output_dir=True,
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
        remove_unused_columns=True,
        save_steps=500,
        seed=0,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        warmup_ratio=0.2,
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
    trainer.log_metrics("train", train_result.metrics)

    mlflow.transformers.log_model(
        transformers_model={"model": trainer.model, "tokenizer": tokenizer},
        artifact_path=output_dir,
    )

    tokenizer.padding_side = 'left'
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(test_dataset)
    trainer.log_metrics("eval", eval_metrics)

def main(train_file, eval_file, model_output_dir):
    """
    Main function to fine-tune the model.
    """
    model_kwargs = {
        "use_cache": False,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": None,
        "attn_implementation": "eager"
    }

    # pretrained_model_name = "microsoft/Phi-3-mini-4k-instruct"
    pretrained_model_name = "microsoft/Phi-3.5-mini-instruct"

    with mlflow.start_run():
        model, tokenizer = initialize_model_and_tokenizer(pretrained_model_name, model_kwargs)
        train_dataset, test_dataset = load_and_preprocess_data(train_file, eval_file, tokenizer)
        train_and_evaluate_model(train_dataset, test_dataset, model, tokenizer, model_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, required=True, help="Path to the training data")
    parser.add_argument("--eval-file", type=str, required=True, help="Path to the evaluation data")
    parser.add_argument("--model_output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    args = parser.parse_args()
    main(args.train_file, args.eval_file, args.model_output_dir)
