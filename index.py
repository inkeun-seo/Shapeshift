import asyncio
import os
from typing import Dict, List, Any, Union, Literal
import aiohttp
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

EmbeddingClient = Literal['cohere', 'openai', 'voyage']

class Shapeshift:
    def __init__(self, embedding_client: EmbeddingClient, api_key: str, 
                 embedding_model: str = '', similarity_threshold: float = 0.5):
        self.embedding_client = embedding_client
        self.api_key = api_key
        self.similarity_threshold = similarity_threshold

        if embedding_client == 'cohere':
            self.embedding_model = embedding_model or 'embed-english-v3.0'
        elif embedding_client == 'openai':
            self.embedding_model = embedding_model or 'text-embedding-ada-002'
        elif embedding_client == 'voyage':
            self.embedding_model = embedding_model or 'voyage-large-2'
        else:
            raise ValueError('Unsupported embedding client')

        # For local embedding fallback
        self.local_model = SentenceTransformer('all-MiniLM-L6-v2')

    async def calculate_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Calculating embeddings for {len(texts)} texts using {self.embedding_client} client")
        if self.embedding_client == 'cohere':
            return await self._cohere_embeddings(texts)
        elif self.embedding_client == 'openai':
            return await self._openai_embeddings(texts)
        elif self.embedding_client == 'voyage':
            return await self._voyage_embeddings(texts)
        else:
            # Fallback to local embedding
            return self.local_model.encode(texts).tolist()

    async def _cohere_embeddings(self, texts: List[str]) -> List[List[float]]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.cohere.ai/v1/embed',
                json={'texts': texts, 'model': self.embedding_model},
                headers={'Authorization': f'Bearer {self.api_key}'}
            ) as response:
                data = await response.json()
                logger.info(f"Received embeddings from Cohere: {data}")
                return data['embeddings']

    async def _openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        async with aiohttp.ClientSession() as session:
            embeddings = []
            for text in texts:
                async with session.post(
                    'https://api.openai.com/v1/embeddings',
                    json={'input': text, 'model': self.embedding_model},
                    headers={'Authorization': f'Bearer {self.api_key}'}
                ) as response:
                    data = await response.json()
                    logger.info(f"Received embeddings from OpenAI for text: {text}")
                    embeddings.append(data['data'][0]['embedding'])
            return embeddings

    async def _voyage_embeddings(self, texts: List[str]) -> List[List[float]]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.voyageai.com/v1/embeddings',
                json={'input': texts, 'model': self.embedding_model},
                headers={'Authorization': f'Bearer {self.api_key}'}
            ) as response:
                data = await response.json()
                logger.info(f"Received embeddings from Voyage: {data}")
                return [item['embedding'] for item in data['data']]

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    def find_closest_match(self, source_embedding: List[float], target_embeddings: List[List[float]]) -> Union[int, None]:
        similarities = [self.cosine_similarity(source_embedding, target_embedding) for target_embedding in target_embeddings]
        max_similarity = max(similarities)
        if max_similarity >= self.similarity_threshold:
            return similarities.index(max_similarity)
        return None

    @staticmethod
    def flatten_object(obj: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        flattened = {}
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flattened.update(Shapeshift.flatten_object(v, key))
            else:
                flattened[key] = v
        return flattened

    @staticmethod
    def unflatten_object(obj: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in obj.items():
            parts = key.split('.')
            d = result
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value
        return result

    async def shapeshift(self, source_obj: Dict[str, Any], target_obj: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Flattening source and target objects")
        flattened_source_obj = self.flatten_object(source_obj)
        flattened_target_obj = self.flatten_object(target_obj)

        source_keys = list(flattened_source_obj.keys())
        target_keys = list(flattened_target_obj.keys())

        logger.info(f"Calculating embeddings for source keys: {source_keys}")
        source_embeddings = await self.calculate_embeddings(source_keys)
        logger.info(f"Calculating embeddings for target keys: {target_keys}")
        target_embeddings = await self.calculate_embeddings(target_keys)

        flattened_result = {}

        for i, source_key in enumerate(source_keys):
            source_embedding = source_embeddings[i]
            closest_target_index = self.find_closest_match(source_embedding, target_embeddings)
            
            if closest_target_index is not None:
                closest_target_key = target_keys[closest_target_index]
                flattened_result[closest_target_key] = flattened_source_obj[source_key]
                logger.info(f"Mapping source key {source_key} to target key {closest_target_key}")

        return self.unflatten_object(flattened_result)

# Example usage
async def main():
    source_obj = {
        "personalInfo": {
            "name": "John Doe",
            "age": 30,
        },
        "occupation": "Software Engineer",
        "FullAddress": "123 Main St, Anytown",
        "address": {
            "street": "123 Main St",
            "city": "Anytown"
        }
    }

    target_obj = {
        "fullName": "",
        "yearsOld": 0,
        "profession": "",
        "location": {
            "streetAddress": "",
            "cityName": ""
        }
    }

    try:
        # Cohere example
        # shapeshifter = Shapeshift('cohere', os.getenv('COHERE_API_KEY', ''))
        # shifted_obj = await shapeshifter.shapeshift(source_obj, target_obj)
        # logger.info(f"Cohere Shifted object: {shifted_obj}")

        # OpenAI example
        openai_shapeshifter = Shapeshift('openai', os.getenv('OPENAI_API_KEY', ''))
        openai_shifted_obj = await openai_shapeshifter.shapeshift(source_obj, target_obj)
        logger.info(f"OpenAI Shifted object: {openai_shifted_obj}")

        # Voyage example
        # voyage_shapeshifter = Shapeshift('voyage', os.getenv('VOYAGE_API_KEY', ''))
        # voyage_shifted_obj = await voyage_shapeshifter.shapeshift(source_obj, target_obj)
        # logger.info(f"Voyage Shifted object: {voyage_shifted_obj}")

    except Exception as error:
        logger.error(f"Error: {str(error)}")

if __name__ == "__main__":
    asyncio.run(main())
