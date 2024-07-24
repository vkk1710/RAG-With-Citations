from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

from vectordb.base_vectordb import BaseVectordb

class QdrantDB(BaseVectordb):
    def __init__(self, collection=""):
        self.client = QdrantClient("localhost", port=6333)
    
    def create_collection(self, collection_name:str, vector_emb:np.ndarray, documents):
        encoder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

        collection = self.client.recreate_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=encoder.get_sentence_embedding_dimension(),
                            distance=Distance.COSINE,
                        ),
                    )
        
        # docs = [
        #     {
        #         "text": doc
        #     }
        #     for doc in documents
        # ]

        self.client.upload_points(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=idx, vector=vector_emb[idx].tolist(), payload=doc
                )
                for idx, doc in enumerate(documents)
            ],
        )
    
    def search(self, collection_name, query_emb:np.ndarray, topk:int, **kwargs) -> List[str]:
        hits = self.client.search(
            collection_name=collection_name,
            query_vector=query_emb[0].tolist(),
            limit=topk
        )
        
        result = [hit.payload for hit in hits]

        return result



