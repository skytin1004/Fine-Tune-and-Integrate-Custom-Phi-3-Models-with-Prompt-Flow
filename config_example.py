# Azure settings
AZURE_SUBSCRIPTION_ID = "your_subscription_id"
AZURE_RESOURCE_GROUP_NAME = "your_resource_group_name" # "TestGroup"

# Azure Machine Learning settings
AZURE_ML_WORKSPACE_NAME = "your_workspace_name" # "finetunephi-workspace"

# Azure Managed Identity settings
AZURE_MANAGED_IDENTITY_CLIENT_ID = "your_azure_managed_identity_client_id"
AZURE_MANAGED_IDENTITY_NAME = "your_azure_managed_identity_name" # "finetunephi-mangedidentity"
AZURE_MANAGED_IDENTITY_RESOURCE_ID = f"/subscriptions/{AZURE_SUBSCRIPTION_ID}/resourceGroups/{AZURE_RESOURCE_GROUP_NAME}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/{AZURE_MANAGED_IDENTITY_NAME}"

# Dataset file paths
TRAIN_DATA_PATH = "data/train_data.jsonl"
TEST_DATA_PATH = "data/test_data.jsonl"

# Fine tuned model settings
AZURE_MODEL_NAME = "your_fine_tuned_model_name" # "finetune-phi-model"
AZURE_ENDPOINT_NAME = "your_fine_tuned_model_endpoint_name" # "finetune-phi-endpoint"
AZURE_DEPLOYMENT_NAME = "your_fine_tuned_model_deployment_name" # "finetune-phi-deployment"

AZURE_ML_API_KEY = "your_fine_tuned_model_api_key"
AZURE_ML_ENDPOINT = "your_fine_tuned_model_endpoint_uri" # "https://{your-endpoint-name}.{your-region}.inference.ml.azure.com/score"