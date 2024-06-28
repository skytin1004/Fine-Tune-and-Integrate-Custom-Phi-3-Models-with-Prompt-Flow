import logging
import requests
from promptflow.core import tool
import asyncio
import platform
from config import (
    AZURE_ML_ENDPOINT,
    AZURE_ML_API_KEY
)

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

def query_azml_endpoint(input_data: list, endpoint_url: str, api_key: str) -> str:
    """
    Send a request to the Azure ML endpoint with the given input data.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "input_data": [input_data],
        "params": {
            "temperature": 0.7,
            "max_new_tokens": 128,
            "do_sample": True,
            "return_full_text": True
        }
    }
    try:
        response = requests.post(endpoint_url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()[0]
        logger.info("Successfully received response from Azure ML Endpoint.")
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying Azure ML Endpoint: {e}")
        raise

def setup_asyncio_policy():
    """
    Setup asyncio event loop policy for Windows.
    """
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        logger.info("Set Windows asyncio event loop policy.")

@tool
def my_python_tool(input_data: str) -> str:
    """
    Tool function to process input data and query the Azure ML endpoint.
    """
    setup_asyncio_policy()
    return query_azml_endpoint(input_data, AZURE_ML_ENDPOINT, AZURE_ML_API_KEY)
