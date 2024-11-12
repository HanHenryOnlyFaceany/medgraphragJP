import numpy as np
from typing import List, Optional
from openai import AsyncOpenAI,Stream
from camel.models import OpenSourceModel
from camel.embeddings import OpenSourceEmbedding
from camel.types import ModelType, EmbeddingModelType
from camel.types import ChatCompletion
from camel.messages import OpenAIMessage
from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
import requests


def clean_message_data(msg):
    # Example of handling potential Union types in msg
    if isinstance(msg.get("content"), (str, dict)):
        # Handle as required, for instance convert dict to string if needed
        msg["content"] = str(msg["content"]) if isinstance(msg["content"], dict) else msg["content"]
    return msg

# Text generation model configuration for Llama 3.2
model_config = {
    "model_path": "hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/llama-3.2-1b-instruct-q8_0.gguf",
    "server_url": "http://host.docker.internal:1234/v1",
    "api_params": {
        "temperature": 0.7,
        "max_tokens": 1200,
    },
}

# Embedding model configuration for Mistral
embedding_config = {
    "model_path": "gaianet/Nomic-embed-text-v1.5-Embedding-GGUF/nomic-embed-text-v1.5.f16.gguf",
    "server_url": "http://host.docker.internal:1234/v1",
    "api_params": {
        "temperature": 0.0,
        "max_tokens": 1000,
    },
}
# Set model type for Llama 3.2
model_type = ModelType.LLAMA_3

model_type_embed = ModelType.NOMIC_EMBED
# Initialize OpenSourceModel for text generation
open_source_model = OpenSourceModel(model_type=model_type, model_config_dict=model_config)

open_source_embed = OpenSourceEmbedding(model_type=model_type_embed, model_config_dict=embedding_config)

async def openai_complete_if_cache(
    model, prompt: str, system_prompt: Optional[str] = None, history_messages: List[dict] = [], **kwargs
) -> str:
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    messages_openai = [messages]



    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    # Run model inference using OpenSourceModel
    response = open_source_model.run(messages_openai)
    if isinstance(response, ChatCompletion):
        response_text = response.choices[0].message.content
    elif isinstance(response, Stream):
        response_text = "".join(chunk.choices[0].message.content for chunk in response)
    else:
        raise TypeError("Unexpected response type")


    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response_text, "model": model}}
        )

    return response_text

async def gpt_4o_complete(
    prompt: str, system_prompt: Optional[str] = None, history_messages: List[dict] = [], **kwargs
) -> str:
    return await openai_complete_if_cache(
        open_source_model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

async def gpt_4o_mini_complete(
    prompt: str, system_prompt: Optional[str] = None, history_messages: List[dict] = [], **kwargs
) -> str:
    return await openai_complete_if_cache(
        open_source_model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

# @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
# async def openai_embedding(texts: list[str]) -> np.ndarray:
#     openai_async_client = AsyncOpenAI()
#     response = await openai_async_client.embeddings.create(
#         model="text-embedding-3-small", input=texts, encoding_format="float"
#     )
#     return np.array([dp.embedding for dp in response.data])

@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192)
async def openai_embedding(texts: List[str]) -> np.ndarray:
    # Generate embeddings using the OpenSourceEmbedding class
    embeddings = open_source_embed.embed_list(texts)
    
    # Convert the list of embeddings to a numpy array and return
    return np.array([dp.embedding for dp in embeddings])