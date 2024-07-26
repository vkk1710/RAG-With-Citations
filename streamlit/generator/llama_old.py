import transformers
import torch
from typing import List
import re

from .base_generator import BaseGenerator

class LlamaGenerator(BaseGenerator):
    def __init__(self, model_name, token) -> None:
        self.pipeline = transformers.pipeline(
                                                "text-generation",
                                                model=model_name,
                                                model_kwargs={"torch_dtype": torch.float16},
                                                device_map="auto",
                                                # device="cuda:0",
                                                token=token
                                            )
        self.terminators = [
                        self.pipeline.tokenizer.eos_token_id,
                        self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
        
        self.max_length  = 8192

    def build_prompt(self, user_query: str, contexts: List[str], chat_history:List[str]=[]) -> str:
        messages =  [
                        {"role": "system",
                          "content": """You are an automotive expert specialized in repair and maintenance of a vehicle. Answer the questions asked by a car owner or a technician to assist with repair and maintenance. Summarize the issue, resolution in a descriptive manner to communicate with a customer or technician. Use the contexts provided below and answer the question following the contexts. The answer should be generated using the contexts only. If the contexts seems insufficient to answer the question respond with a message stating that question cannot be asnwered due to lack of information. 
                          
                        Remember, you must return both an answer and citations. A citation consists of a VERBATIM quote that \
                        justifies the answer and the ID of the quote article. Return a citation for every quote across all articles \
                        that justify the answer. Return the final output in the exact json format below:
                        
                        {
                            "answer": "911 Assist is a SYNC system feature that can call for help in the event of a crash. It works by using a paired and connected Bluetooth-enabled phone to dial 911 if a crash deploys an airbag, excluding knee airbags and rear inflatable seatbelts, or activates the fuel pump shut-off. The system transmits vehicle data to the emergency service during an emergency call.",
                            "citations": [
                                {
                                    "source_id": 1,
                                    "quote": "911 Assist is a SYNC system feature that can call for help."
                                }
                                {
                                    "source_id": 2,
                                    "quote": "If a crash deploys an airbag, excluding knee airbags and rear inflatable seatbelts, 
                                    or activates the fuel pump shut-off, your vehicle may be able to contact emergency services by dialing 911 through a paired 
                                    and connected Bluetooth-enabled phone."
                                },
                                {
                                    "source_id": 4,
                                    "quote": "During an emergency call the system transmits vehicle data to the emergency service."
                                }
                                ...
                            ]
                        }

                        """
                          },
                        
                    ]
        
        # Use the following format for your final output:
        # <cited_answer>
        #                     <answer></answer>
        #                     <citations>
        #                         <citation><source_id></source_id><quote></quote></citation>
        #                         <citation><source_id></source_id><quote></quote></citation>
        #                         ...
        #                     </citations>
        #                 </cited_answer>

        query = "Contexts:\n"
        query += "\n".join([f'{i+1}. {context}' for i,context in enumerate(contexts)])
        
        if chat_history:
            query += '\n Below are the questions previously asked by the user.\n'

            while True:
                prev_questions = '\n'.join([f'{i+1}. {chat}' for i,chat in enumerate(chat_history)])
                temp = query + prev_questions
                temp += f'\nQuestion: {user_query}'

                print(temp)

                temp_message = messages + [ {"role": "user", "content": temp} ]

                prompt = self.pipeline.tokenizer.apply_chat_template(
                                                            temp_message, 
                                                            tokenize=False, 
                                                            add_generation_prompt=True
                                                                    )
                
                if len(prompt) < self.max_length:
                    break

                else:
                    chat_history.pop(0)

        else:
            query += f'\nQuestion: {user_query}'

            messages.append( {"role": "user", "content": query} )


            prompt = self.pipeline.tokenizer.apply_chat_template(
                                                                messages, 
                                                                tokenize=False, 
                                                                add_generation_prompt=True
                                                                        )
        
        return prompt, chat_history
    
    def parse_response(self, model_output: str) -> str:
        response = model_output.split("<|end_header_id|>")[-1]
        return response.strip()
    
    def generate(self, query: str, contexts: List[str], chat_history:List[str]=[], **pipeline_kwargs) -> str:
        prompt, chat_history = self.build_prompt(user_query=query, contexts=contexts, chat_history=chat_history)
        model_outputs = self.pipeline(
                            prompt,
                            eos_token_id=self.terminators,
                            **pipeline_kwargs
                        )
        model_outputs = model_outputs[0]['generated_text']
        response = self.parse_response(model_output=model_outputs)

        return {
                'response':response,
                'chat_history':chat_history
            }
        
        
