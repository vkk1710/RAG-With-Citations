from typing import List
import numpy as np

class BaseRetriever:
    def embed(self, query:List[str], **kwargs) -> np.ndarray:
        raise NotImplementedError('Please implement this method.')
    def retrieve(self, query:List[str], **kwargs) -> List[str]:
        raise NotImplementedError('Please implement this method.')