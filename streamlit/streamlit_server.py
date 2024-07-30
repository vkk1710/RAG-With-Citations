from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import json
import re
import uvicorn

from retrieval.mxbai_retriever import MxbaiRetriever
from reranker.mxbai_reranker import MxbaiReranker
from generator.llama_old import LlamaGenerator
from vectorDB.qdrantdb import QdrantDB
from highlight_docs import *
from typing import List

db = QdrantDB()
collection_name = 'test2'

retriever_model = 'intfloat/e5-large'
# retriever_model = 'mixedbread-ai/mxbai-embed-large-v1'
retriever = MxbaiRetriever(retriever_model,db)
ranker = MxbaiReranker('mixedbread-ai/mxbai-rerank-large-v1')
generator = LlamaGenerator('meta-llama/Meta-Llama-3.1-8B-Instruct', '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')

embedder_prompts = {
    'mixedbread-ai/mxbai-embed-large-v1': {
        'query_prompt': 'Represent this sentence for searching relevant passages: ',
        'passage_prompt': None
    },
    'intfloat/e5-large': {
        'query_prompt': 'query: ',
        'passage_prompt': 'passage: '
    }
}

app = FastAPI()

class LLMInput(BaseModel):
    query: str
    temperature: float
    top_p: float
    max_new_tokens: int
    chat_history: List[str]
    

def output_formatter(raw_output):
    output = {
        "answer": "",
        "citations": []
    }
    
    citations_list = []
    out = ' '.join(raw_output.splitlines())
    content = re.split(r'"?[A|a]nswer"?:', out)[1]
    content = re.split(r'"?[C|c]itations"?:', content)
    answer_text = content[0].strip()
    citations_text = content[1]
    citations_text = re.split(r'"?[N|n]ote"?:', citations_text)[0].strip()
    
    citations = re.split(r'\n', citations_text)
    
    for cit in citations:
        cit_text = ''
        cit_index = ''
        
        match = re.search(""" ["|'](.*)["|'] """, cit)
        if match != None:
            cit_text = match.group(1).strip()
        else:
            match = re.split("""\\([S|s]ource[ |\\-|\\_][I|i][D|d]""", cit)[0].strip()
            match = re.sub("^\\W", "", match)
            match = re.sub("\\W$", "", match)
            cit_text = match.strip()
            
        match = re.search("""\\([S|s]ource[ |\\-|\\_][I|i][D|d][\\:|\\-]? (\\d+)""", cit)
        if match != None:
            cit_index = match.group(1).strip()
        else:
            print(f'Cite Index Not Found for Citation-\n{cit}')
            
        if cit_index.isnumeric():
            cit_index = int(cit_index)
        else:
            print(f'Cite Index Is Not INT for Citation-\n{cit}')
            
        cit_dict = {
            "source_id": cit_index,
            "quote": cit_text
        }
        
        citations_list.append(cit_dict)
        
    output['answer'] = answer_text
    output['citations'] = citations_list
    
    return str(output)
    

def clean_output(raw_output):
    is_raw_output = False
    print('\n\nRaw Output from the LLM: ', raw_output, '\n\n')
    output = None
    
    st = re.search('{\n?(\s{1,4})?"[A|a]nswer"', raw_output)
    end = re.search('](\n)?}', raw_output)
    
    if (st == None or end == None):
        match = re.search(r'"?[C|c]itations"?:', raw_output)
        if match == None:
            output = raw_output
            is_raw_output = True
        
        else:
            raw_output = output_formatter(raw_output)
            output = json.loads(raw_output, strict=False)
            print('The output was not a JSON. Fixed it.')
        return output, is_raw_output

    st = st.start()
    end = end.end()
    
    if (st<0 or end<0 or st > end):
        output = raw_output
        is_raw_output = True
        print('JSON structure is not correct!')
    else:
        raw_output = raw_output[st:end]
        output = json.loads(raw_output, strict=False)
        
    return output, is_raw_output


@app.post('/chat')
def llm_chat(params: LLMInput):
    params = params.json()
    params = json.loads(params)

    query = params['query']
    temperature = params['temperature']
    top_p = params['top_p']
    max_new_tokens = params['max_new_tokens']
    chat_history = params['chat_history']
    query_prompt = embedder_prompts[retriever_model]['query_prompt']

    result = retriever.retrieve(collection_name=collection_name, query=[query], topk=10, prompt=query_prompt)
    if len(result) == 0:
        response = {
            'answer': 'No context available to answer this question!',
            'highlighted_docs': [],
            'chat_history': []
        }
        return response
    if retriever_model == 'mixedbread-ai/mxbai-embed-large-v1':
        ranked_result = ranker.rank(query, result, topk=10)
    else:
        ranked_result = result
    llm_context = [res['text'] for res in ranked_result]
    answer = generator.generate(query, llm_context, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature)
    
    print('\n\nRetrieved Result before ranking: ',json.dumps(result, indent=2), '\n\n')
    print('\n\nRanked result after re-ranking of retrieved data: ',json.dumps(ranked_result, indent=2))
 
    output, is_raw_output = clean_output(answer['response'])
    chat_history = answer['chat_history']
    
    highlighted_pdfs = []
    highlighted_csvs = []
    
    if not is_raw_output:
        final_indices = validate_output(llm_context, output)
        highlights_list = get_citations(final_indices, ranked_result)

        pdf_highlights_list = [data for data in highlights_list if data['file_name'].split('.')[-1] == 'pdf']
        csv_highlights_list = [data for data in highlights_list if data['file_name'].split('.')[-1] == 'csv']

        if len(pdf_highlights_list) > 0:
            highlighted_pdfs = highlight_pdf(pdf_highlights_list)
            
        if len(csv_highlights_list) > 0:
            highlighted_csvs = highlight_csv(csv_highlights_list)
    
    highlighted_docs = highlighted_pdfs+highlighted_csvs
    
    # print(highlighted_docs)
    
    response = {
        'answer': output,
        'highlighted_docs': highlighted_docs,
        'chat_history': chat_history
    }
    
    return response

if __name__ == "__main__":
    uvicorn.run(host='0.0.0.0',port=8003,app=app)