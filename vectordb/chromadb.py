import chromadb
import numpy as np
from typing import List

from .base_vectordb import BaseVectordb

class ChromaDB(BaseVectordb):
    def __init__(self, collection=""):
        self.client = chromadb.PersistentClient(path='./chroma_db_collections')
    
    def create_collection(self, collection_name:str, vector_emb:np.ndarray, documents:List[str]):

        if collection_name in self.client.list_collections():
            self.client.delete_collection(collection_name)

        collection = self.client.create_collection(name=collection_name)
        collection.add(
            embeddings = vector_emb.tolist(),
            documents=documents,
            ids = [f'ids_{i}' for i in range(len(documents))]
        )
    
    def search(self, collection_name, query_emb:np.ndarray, topk:int, **kwargs) -> List[str]:
        collection = self.client.get_collection(name=collection_name)
        result = collection.query(
            query_embeddings=query_emb,
            n_results=topk
        )

        return result['documents']



