from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import torch

from .base_retrieval import BaseRetriever

class MxbaiRetriever(BaseRetriever):
    def __init__(self, model_name, vectordb_client):
        self.model = SentenceTransformer(model_name)
        self.vectordb_client = vectordb_client

    def embed(self, query:List[str], **kwargs) -> np.ndarray:
        emb = self.model.encode( query, show_progress_bar=False, convert_to_numpy=True, **kwargs)
        # pool = self.model.start_multi_process_pool()
        # emb = self.model.encode_multi_process(query, pool)
        # self.model.stop_multi_process_pool(pool)
        # print(f'encoder device = {self.model.device}')
        return emb

    def retrieve(self, collection_name, query: List[str], **kwargs) -> List[str]:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        prompt = kwargs.get('prompt', None)
        topk = kwargs.get('topk', 10)
        print('topk: ', topk)
        print('prompt: ', prompt)
        
        query_emb = self.embed(query=query, batch_size=32, device=device, prompt=prompt)
        contexts = self.vectordb_client.search(collection_name=collection_name, query_emb=query_emb, topk=topk)
        return contexts
        