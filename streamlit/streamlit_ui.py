import streamlit as st
import json
import requests
import os
import fitz
import time
import pandas as pd
from vectorDB.qdrantdb import QdrantDB
from chunking.sentence_chunker import PDFSentenceChunker, CSVDataLoader, TSBSentenceChunker
from retrieval.mxbai_retriever import MxbaiRetriever
from streamlit_pdf_viewer import pdf_viewer

st.title("Predii RAG")

#Load config
config_path = "./config.json"

with open(config_path, 'r') as j:
     config = json.loads(j.read())

collection_name = config["collection_name"]
retriever_model = config["retriever_model"]
is_TSB = config["is_TSB"]

start_time = time.time()
client = QdrantDB(is_TSB=is_TSB) 

retriever = MxbaiRetriever(retriever_model, client)
end_time = time.time()
print(f'Created the Qdrant client and the retriever object in {end_time - start_time} secs')

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
    
def upload_docs(docs_path='./streamlit_documents', is_TSB = True):
    file_names = []
    start_time = time.time()
    if is_TSB:
        chunks = TSBSentenceChunker(file_dir=docs_path, document_names=file_names).chunk()
    else:
        chunks = PDFSentenceChunker(file_dir=docs_path, document_names=file_names).chunk_new() + CSVDataLoader(file_dir=docs_path, document_names=file_names).load_data()
    
    end_time = time.time()
    print(f'Time for chunking = {end_time - start_time} secs')
    
    chunk_sents = [chunk['text'] for chunk in chunks]
    
    if is_TSB:
        chunk_metadata = list(set([(chunk['metadata'], chunk['file_name']) for chunk in chunks]))
        chunk_meta_texts = [x[0] for x in chunk_metadata]
    
    # print(f'Number of chunk sentences = {len(chunk_sents)}\n')
    # print('Chunks after chunking done = \n\n', chunks[:5], '\n')
    
    passage_prompt = embedder_prompts[retriever_model]['passage_prompt']
    
    start_time = time.time()
    emb = retriever.embed(chunk_sents, prompt=passage_prompt)
    
    if is_TSB:
        meta_emb = retriever.embed(chunk_meta_texts, prompt=passage_prompt)    
        meta_emb = {chunk_metadata[idx][1]:meta_emb[idx] for idx, _ in enumerate(chunk_metadata)}
    # print('emb dimensions = ', emb.shape, '\n')
    # print('meta_emb = ', meta_emb, '\n')
    
    end_time = time.time()
    print(f'Time for embedding = {end_time - start_time} secs')
        
    start_time = time.time()
    client.create_collection(collection_name, retriever_model)
    st.toast('Created new collection!')
    
    if is_TSB:
        vectors = {
            'text': emb,
            'metadata': meta_emb
        }
    else:
        vectors = {
            'text': emb,
            'metadata': None
        }
     
    client.upload_points(collection_name, vectors, chunks)
    end_time = time.time()
    print(f'Time for uploading all the points in the collection = {end_time - start_time} secs')
    
    st.toast('Uploaded the documents to Vector DB!')
        
def display_citations(highlighted_docs):    
    docs_path = './streamlit_documents/highlighted_docs'
    for doc in highlighted_docs:
        st.markdown(f'\n-- {doc[0]} \n\tPages - {doc[1]}\n\n')
        file_path = os.path.join(docs_path, doc[0])
        if doc[0].split('.')[-1] == 'pdf':
            pdf_viewer(file_path, pages_to_render=doc[1])
            
        else:
            df = pd.read_excel(file_path)
            st.dataframe(df)
            

with st.sidebar:
    docs_source = './streamlit_documents'
    upload_docs(docs_source, is_TSB)
    st.markdown("""---""")
    temp = st.sidebar.slider('Temperature', 0.0, 1.0, 0.1)
    max_tokens = st.sidebar.slider('Max New Tokens', 0, 8192, 1024)
    top_p = st.sidebar.slider('Top P', 0.0, 1.0, 0.9)
    
        

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter Query"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    data = {
        'query':prompt,
        'temperature':temp,
        'max_new_tokens':max_tokens,
        'top_p':top_p,
        'chat_history':st.session_state.chat_history
    }

    data = json.dumps(data)

    output = requests.post(url='http://127.0.0.1:8003/chat', data=data).text
    output = json.loads(output)
    
    response = output['answer']
    highlighted_docs = output['highlighted_docs']
    chat_history = output['chat_history']
    
    if len(highlighted_docs):
        response = response['answer']
    
    with st.chat_message("assistant"):
        st.markdown(response)
        if len(highlighted_docs):
            st.markdown("\nREFERENCES - \n")
            # st.markdown(f'{highlighted_docs}')
            display_citations(highlighted_docs)
        

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role":"assistant", "content": response})

    chat_history.append(prompt)

    st.session_state.chat_history = chat_history