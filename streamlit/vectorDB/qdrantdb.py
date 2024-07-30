from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, FilterSelector, Filter, FieldCondition, MatchValue, MatchAny
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

import streamlit as st

from vectorDB.base_vectordb import BaseVectordb

class QdrantDB(BaseVectordb):
    def __init__(self, collection="", is_TSB = True):
        # self.client = QdrantClient(path="./qdrant_db_collections")
        self.client = QdrantClient("localhost", port=6333)
        self.is_TSB = is_TSB
    
    def create_collection(self, collection_name:str, embedder_model:str):
        encoder = SentenceTransformer(embedder_model)
        
        if self.is_TSB:
            collection = self.client.recreate_collection(
                            collection_name=collection_name,
                            vectors_config={
                                    "metadata": VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE),
                                    "text": VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE),
                                },
                        )
        else:
            collection = self.client.recreate_collection(
                            collection_name=collection_name,
                            vectors_config={
                                    "text": VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE),
                                },
                        )
        
    def upload_points(self, collection_name, vector_embs, documents):
        # print('documents in upload_points func = \n', documents, '\n\n')
        # print('vector_embs in upload_points func = \n', vector_emb, '\n\n')
        # print('meta_vector_emb in upload_points func = \n', meta_vector_emb, '\n\n')
        vector_emb = vector_embs['text']
        meta_vector_emb = vector_embs['metadata']
        # print('Text Vectors: ', vector_embs, '\n\n')   
        # print('Text Vectors: ', vector_emb, '\n\n')   
        
        if self.is_TSB:
            self.client.upload_points(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=idx, 
                        vector={
                                "metadata": meta_vector_emb[doc['file_name']].tolist(),
                                "text": vector_emb[idx].tolist()
                            }, 
                        payload=doc
                    )
                    for idx, doc in enumerate(documents)
                ],
            )
        else:
            self.client.upload_points(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=idx, 
                        vector={
                                "text": vector_emb[idx].tolist()
                            }, 
                        payload=doc
                    )
                    for idx, doc in enumerate(documents)
                ],
            )
        # st.session_state.points_idx += len(documents)
        # st.toast(f'No of points at end of upload in class ---- {st.session_state.points_idx}')
    
    def search(self, collection_name, query_emb:np.ndarray, topk:int, **kwargs) -> List[str]:
        query = query_emb[0].tolist()
        
        if not self.is_TSB:
            hits = self.client.search(
                collection_name=collection_name,
                query_vector=("text", query),
                limit=topk
            )
        
        else:
            meta_hits = self.client.search(
                collection_name=collection_name,
                query_vector=("metadata", query),
                limit=100
            )
            
            meta_result = list(set([hit.payload['metadata'] for hit in meta_hits]))
            print(f'No of unique metadata fetched by the Retriever: {len(meta_result)}\n')
            
            hits = self.client.search(
                collection_name=collection_name,
                query_vector=("text", query),
                query_filter=Filter(
                    must=[
                        FieldCondition(key="metadata", match=MatchAny(any=meta_result)),
                    ]
                ),
                limit=topk
            )
        
        result = [hit.payload for hit in hits]

        return result


    def delete_points(self, collection_name, documents: List[str]):
        try:
            for doc in documents:
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="file_name",
                                    match=MatchValue(value=doc),
                                ),
                            ],
                        )
                    ),
                )
        except Exception as err:
            print('There has been an error while deleting the points in QDrant DB. \nError: \n{err}')