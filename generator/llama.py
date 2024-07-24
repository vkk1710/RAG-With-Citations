import torch
import re
import requests
import json
from typing import List

from .base_generator import BaseGenerator

class LlamaGenerator(BaseGenerator):
    def __init__(self) -> None:
        self.input_data = {
            "prompt": "",
            "temperature": 0.1,
            "max_tokens": 1024,
            "top_p": 1,
            "top_k": -1
        }
        
        self.url = "http://127.0.0.1:8081/vllm/models/generate"
        
        self.max_length  = 8192

    def build_prompt(self, user_query: str, contexts: List[str], chat_history:List[str]=[]) -> str:
        messages =  [   """
                       You are a bot specializing in Automotive Domain. Use the contexts provided below and answer the question following the contexts. The answer should be generated using the contexts only. If the contexts seems insufficient to answer the question respond with a message stating that question cannot be asnwered due to lack of information. Remember, you must return both an answer and citations. A citation consists of a VERBATIM quote that \
                        justifies the answer and the ID of the quote article. Remember the ID of the quote articles start from 0. Return a citation for every quote across all articles \
                        that justify the answer. Use the following format for your final output:

                        <cited_answer>
                            <answer></answer>
                            <citations>
                                <citation><source_id></source_id><quote></quote></citation>
                                <citation><source_id></source_id><quote></quote></citation>
                                ...
                            </citations>
                        </cited_answer>
                     """   
                    ]

        query = "Contexts:\n"
        query += "\n".join([f'{i+1}. {context}' for i,context in enumerate(contexts)])
        
        if chat_history:
            query += '\n Below are the questions previously asked by the user.\n'

            while True:
                prev_questions = '\n'.join([f'{i+1}. {chat}' for i,chat in enumerate(chat_history)])
                temp = query + prev_questions
                temp += f'\nQuestion: {user_query}'
                query += "\nAnswer: "

                print(temp)

                temp_message = messages + [temp]
                prompt = '\n\n'.join(temp_message)
                
                if len(prompt) < self.max_length:
                    break

                else:
                    chat_history.pop(0)

        else:
            query += f'\nQuestion: {user_query}'
            query += "\nAnswer: "

            messages.append(query)


            prompt = '\n'.join(messages)
        
        return prompt, chat_history
    
    def generate(self, query: str, contexts: List[str], chat_history:List[str]=[], **pipeline_kwargs) -> str:
        prompt, chat_history = self.build_prompt(user_query=query, contexts=contexts, chat_history=chat_history)
        self.input_data["prompt"] = prompt
        data = json.dumps(self.input_data)
        
        output = requests.post(url=self.url, data=data)
        response = eval(output.text)
        
        print(prompt)

        return {
                'response':response,
                'chat_history':chat_history
            }
        
        
