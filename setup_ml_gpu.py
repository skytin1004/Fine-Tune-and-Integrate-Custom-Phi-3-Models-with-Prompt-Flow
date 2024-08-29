import logging
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import Environment, AmlCompute
from azure.identity import AzureCliCredential
from config import (
    AZURE_SUBSCRIPTION_ID,
    AZURE_RESOURCE_GROUP_NAME,
    AZURE_ML_WORKSPACE_NAME,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH
)

# Constants

# Uncomment the following lines to use a CPU instance for training
# COMPUTE_INSTANCE_TYPE = "Standard_E16s_v3" # cpu
# COMPUTE_NAME = "cpu-e16s-v3"
# DOCKER_IMAGE_NAME = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
# CONDA_FILE = "conda_cpu.yml"

# Uncomment the following lines to use a GPU instance for training
COMPUTE_INSTANCE_TYPE = "Standard_NC6s_v3"
COMPUTE_NAME = "gpu-nc6s-v3"
DOCKER_IMAGE_NAME = "mcr.microsoft.com/azureml/curated/acft-hf-nlp-gpu:59"
CONDA_FILE = "conda_gpu.yml"

LOCATION = "eastus2" # Replace with the location of your compute cluster
FINETUNING_DIR = "./finetuning_dir" # Path to the fine-tuning script
TRAINING_ENV_NAME = "phi-3-training-environment" # Name of the training environment
MODEL_OUTPUT_DIR = "./model_output" # Path to the model output directory in azure ml

# Logging setup to track the process
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.WARNING
)

def get_ml_client():
    """
    Initialize the ML Client using Azure CLI credentials.
    """
    credential = AzureCliCredential()
    return MLClient(credential, AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP_NAME, AZURE_ML_WORKSPACE_NAME)

def create_or_get_environment(ml_client):
    """
    Create or update the training environment in Azure ML.
    """
    env = Environment(
        image=DOCKER_IMAGE_NAME,  # Docker image for the environment
        conda_file=CONDA_FILE,  # Conda environment file
        name=TRAINING_ENV_NAME,  # Name of the environment
    )
    return ml_client.environments.create_or_update(env)

def create_or_get_compute_cluster(ml_client, compute_name, COMPUTE_INSTANCE_TYPE, location):
    """
    Create or update the compute cluster in Azure ML.
    """
    try:
        compute_cluster = ml_client.compute.get(compute_name)
        logger.info(f"Compute cluster '{compute_name}' already exists. Reusing it for the current run.")
    except Exception:
        logger.info(f"Compute cluster '{compute_name}' does not exist. Creating a new one with size {COMPUTE_INSTANCE_TYPE}.")
        compute_cluster = AmlCompute(
            name=compute_name,
            size=COMPUTE_INSTANCE_TYPE,
            location=location,
            tier="Dedicated",  # Tier of the compute cluster
            min_instances=0,  # Minimum number of instances
            max_instances=1  # Maximum number of instances
        )
        ml_client.compute.begin_create_or_update(compute_cluster).wait()  # Wait for the cluster to be created
    return compute_cluster

def create_fine_tuning_job(env, compute_name):
    """
    Set up the fine-tuning job in Azure ML.
    """
    return command(
        code=FINETUNING_DIR,  # Path to fine_tune.py
        command=(
            "python fine_tune.py "
            "--train-file ${{inputs.train_file}} "
            "--eval-file ${{inputs.eval_file}} "
            "--model_output_dir ${{inputs.model_output}}"
        ),
        environment=env,  # Training environment
        compute=compute_name,  # Compute cluster to use
        inputs={
            "train_file": Input(type="uri_file", path=TRAIN_DATA_PATH),  # Path to the training data file
            "eval_file": Input(type="uri_file", path=TEST_DATA_PATH),  # Path to the evaluation data file
            "model_output": MODEL_OUTPUT_DIR
        }
    )

def main():
    """
    Main function to set up and run the fine-tuning job in Azure ML.
    """
    # Initialize ML Client
    ml_client = get_ml_client()

    # Create Environment
    env = create_or_get_environment(ml_client)
    
    # Create or get existing compute cluster
    create_or_get_compute_cluster(ml_client, COMPUTE_NAME, COMPUTE_INSTANCE_TYPE, LOCATION)

    # Create and Submit Fine-Tuning Job
    job = create_fine_tuning_job(env, COMPUTE_NAME)
    returned_job = ml_client.jobs.create_or_update(job)  # Submit the job
    ml_client.jobs.stream(returned_job.name)  # Stream the job logs
    
    # Capture the job name
    job_name = returned_job.name
    print(f"Job name: {job_name}")

if __name__ == "__main__":
    main()
