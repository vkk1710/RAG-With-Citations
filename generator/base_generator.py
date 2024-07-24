from typing import List

class BaseGenerator:

    def build_prompt(self, query:str, contexts:List[str]) -> str:
        raise NotImplementedError('Please Implement this method')
    
    def parse_response(self, model_output:str) -> str:
        raise NotImplementedError('Please Implement this method.')

    def generate(self, query:str, contexts:List[str], **pipeline_kwargs) -> str:
        raise NotImplementedError('Please Implement this method.')