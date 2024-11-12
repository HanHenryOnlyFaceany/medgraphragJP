import requests
from typing import Any, Dict, List, Optional
from camel.embeddings.base import BaseEmbedding
from camel.types import EmbeddingModelType

class Embedding:
    def __init__(self, embedding, index, object_type):
        self.embedding = embedding
        self.index = index
        self.object = object_type

    def __repr__(self):
        return f"Embedding(embedding={self.embedding}, index={self.index}, object='{self.object}')"

# Function to convert the response
def convert_response(response):
    return [
        Embedding(embedding=entry['embedding'], index=entry['index'], object_type=entry['object'])
        for entry in response
    ]

class OpenSourceEmbedding(BaseEmbedding[str]):
    r"""Provides text embedding functionalities using an open-source model served by a local server."""

    def __init__(
        self,
        model_type: EmbeddingModelType,
        model_config_dict: Dict[str, Any],
        server_url: Optional[str] = "http://host.docker.internal:1234/v1",
    ) -> None:
        super().__init__()

        self.model_path = model_config_dict.get("model_path")
        if not self.model_path:
            raise ValueError("Model path is required to interact with the local server.")

        self.server_url = model_config_dict.get("server_url", server_url)
        self.api_params = model_config_dict.get("api_params", {})

        self.output_dim = model_config_dict.get("output_dim", 512)

    def embed_list(self, texts: List[str], **kwargs: Any) -> List[List[float]]:

        payload = {
            "input": texts
        }

        response = requests.post(f"{self.server_url}/embeddings", json=payload)

        if response.status_code != 200:
            raise Exception(f"Error in embedding request: {response.text}")

        embeddings = response.json()['data']
        if embeddings is None:
            raise ValueError("No embeddings returned from the server.")
        embeddings = convert_response(embeddings)
        return embeddings 

    def get_output_dim(self) -> int:
        r"""Returns the output dimension of the embeddings."""
        return self.output_dim
