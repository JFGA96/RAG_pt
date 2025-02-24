import os
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
import tiktoken


dir = os.getcwd()
# Cargar variables de entorno
load_dotenv(dotenv_path=os.path.join(dir,"../.env"))


# Modelo de 4 mini
llm_chat_mini = AzureChatOpenAI(
    azure_deployment = os.getenv('AZURE_OPENAI_GPT4O_MINI'),
    openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION_MINI'),
    api_key = os.getenv('AZURE_OPENAI_API_KEY_MINI'),
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT_MINI'),
    request_timeout = 60,
    max_retries = 2,
    max_tokens=None
)

llm_embed_small = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv('AZURE_OPENAI_EMBED_E_SMALL'),
    openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION_E'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY_E'),
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT_E'),
    chunk_size=1,
    request_timeout=120,
    max_retries=3
)

llm_embed_large = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv('AZURE_OPENAI_EMBED_E_LARGE'),
    openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION_E'),
    api_key=os.getenv('AZURE_OPENAI_API_KEY_E'),
    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT_E'),
    chunk_size=1,
    request_timeout=120,
    max_retries=3
)