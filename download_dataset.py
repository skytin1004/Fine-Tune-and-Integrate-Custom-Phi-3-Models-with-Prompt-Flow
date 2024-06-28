import json
import os
from datasets import load_dataset
from config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH)

def load_and_split_dataset(dataset_name, config_name, split_ratio):
    """
    Load and split a dataset.
    """
    # Load the dataset with the specified name and configuration
    dataset = load_dataset(dataset_name, config_name, split=split_ratio)
    print(f"Original dataset size: {len(dataset)}")
    
    # Split the dataset into train and test sets (80% train, 20% test)
    split_dataset = dataset.train_test_split(test_size=0.2)
    print(f"Train dataset size: {len(split_dataset['train'])}")
    print(f"Test dataset size: {len(split_dataset['test'])}")
    
    return split_dataset

def save_dataset_to_jsonl(dataset, filepath):
    """
    Save a dataset to a JSONL file.
    """
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Open the file in write mode
    with open(filepath, 'w', encoding='utf-8') as f:
        # Iterate over each record in the dataset
        for record in dataset:
            # Dump the record as a JSON object and write it to the file
            json.dump(record, f)
            # Write a newline character to separate records
            f.write('\n')
    
    print(f"Dataset saved to {filepath}")

def main():
    """
    Main function to load, split, and save the dataset.
    """
    # Load and split the dataset with a specific configuration and split ratio
    dataset = load_and_split_dataset("wikitext", 'wikitext-2-v1', 'train[:1%]')
    
    # Extract the train and test datasets from the split
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # Save the train dataset to a JSONL file
    save_dataset_to_jsonl(train_dataset, TRAIN_DATA_PATH)
    
    # Save the test dataset to a separate JSONL file
    save_dataset_to_jsonl(test_dataset, TEST_DATA_PATH)

if __name__ == "__main__":
    main()
