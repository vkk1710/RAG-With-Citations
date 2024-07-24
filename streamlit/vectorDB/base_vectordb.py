from typing import List
import numpy as np

class BaseVectordb:
    def search(self, query_emb:np.ndarray, topk:int, **kwargs) -> List[str]:
        raise NotImplementedError('Please Implement this method.')