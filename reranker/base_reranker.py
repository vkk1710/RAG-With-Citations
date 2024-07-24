from typing import List

class BaseReranker:
    def rank(self, query:str, contexts:List[str], topk:int) -> List[str]:
        raise NotImplementedError('Please Implement this method.')