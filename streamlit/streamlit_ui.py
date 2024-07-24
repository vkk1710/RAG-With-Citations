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

logfile = open("log.txt","w+")

start_time = time.time()
client = QdrantDB() 
collection_name = 'test10'
retriever_model = 'intfloat/e5-large'
# retriever_model = 'mixedbread-ai/mxbai-embed-large-v1'
retriever = MxbaiRetriever(retriever_model, client)
end_time = time.time()
print(f'Created the Qdrant client and the retriever object in {end_time - start_time} secs')
logfile.write(f'Created the Qdrant client and the retriever object in {end_time - start_time} secs\n')

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

def get_all_collections(client):
    try:
        start_time = time.time()
        collections_list = []
        collections = client.client.get_collections()
        for collection in collections:
            for c in list(collection[1]):
                collections_list.append(c.name)
        end_time = time.time()
        print(f'Time for get_all_collections func = {end_time - start_time} secs')
        logfile.write(f'Time for get_all_collections func = {end_time - start_time} secs\n')
        return collections_list
    except Exception as e:
        st.error(f"Error fetching collections from Qdrant: {e}")

def delete_docs(docs_list, source='./streamlit_documents'):
    try:
        for doc in docs_list:
            path = os.path.join(source, doc)
            os.remove(path)
        st.toast('Deleted the selected documents!')
    except Exception as err:
        print(f'There has been an error while deleting the files. \nError: \n{err}')
        
def clear_multi():
    st.session_state['multiselect'] = []
    return


def multi_selector(docs_path='./streamlit_documents'):
    filenames = [file for file in os.listdir(docs_path) if file.split('.')[-1] in ['pdf', 'csv']]
    choices = filenames
    if len(filenames) > 0:
        choices = ['Select All'] + choices
        
    options = st.multiselect("Your Documents:", choices)
    if len(options) > 0:
        res = st.button("Delete", on_click=clear_multi)
        if res:
            if 'Select All' in options:
                options = filenames
            client.delete_points(collection_name, options)
            delete_docs(options)   
    
def upload_docs(docs_path='./streamlit_documents'):
    uploaded_files = st.file_uploader("Upload files (csv or pdf)", accept_multiple_files=True, type=['pdf','csv'])
    file_names = [file.name for file in uploaded_files]
    is_upload = st.button("Upload")
    
    if len(uploaded_files) > 0 and is_upload:
        for file in uploaded_files:
            if file.name in os.listdir(docs_path):
                print(f'{file.name} already exists! So skipping copy.')
                continue
            file_extn = file.name.split('.')[-1]
            output_path = os.path.join(docs_path, file.name)
            
            if file_extn == 'pdf':
                doc = fitz.open(stream=file.read(), filetype="pdf")
                doc.save(output_path, garbage=0, deflate=False, clean=False)
                
            elif file_extn == 'csv':
                df = pd.read_csv(file)
                df.to_csv(output_path)
            
        st.toast('Uploaded the documents!')
        time.sleep(.5)
        st.toast('Uploading to Vector DB...')
        
        start_time = time.time()
        # chunks = PDFSentenceChunker(file_dir=docs_path, document_names=file_names).chunk_new() + CSVDataLoader(file_dir=docs_path, document_names=file_names).load_data()
        chunks = TSBSentenceChunker(file_dir=docs_path, document_names=file_names).chunk()
        end_time = time.time()
        print(f'Time for chunking = {end_time - start_time} secs')
        logfile.write(f'Time for chunking = {end_time - start_time} secs\n')
        
        chunk_sents = [chunk['text'] for chunk in chunks]
        chunk_metadata = list(set([(chunk['metadata'], chunk['file_name']) for chunk in chunks]))
        chunk_meta_texts = [x[0] for x in chunk_metadata]
        
        # print(f'Number of chunk sentences = {len(chunk_sents)}\n')
        # print('Chunks after chunking done = \n\n', chunks[:5], '\n')
        
        passage_prompt = embedder_prompts[retriever_model]['passage_prompt']
        
        start_time = time.time()
        emb = retriever.embed(chunk_sents, prompt=passage_prompt)
        meta_emb = retriever.embed(chunk_meta_texts, prompt=passage_prompt)    
        meta_emb = {chunk_metadata[idx][1]:meta_emb[idx] for idx, _ in enumerate(chunk_metadata)}
        # print('emb dimensions = ', emb.shape, '\n')
        # print('meta_emb = ', meta_emb, '\n')
        
        end_time = time.time()
        print(f'Time for embedding = {end_time - start_time} secs')
        logfile.write(f'Time for embedding = {end_time - start_time} secs\n')
        
        # if not client.client.collection_exists(collection_name=collection_name):
        
        if collection_name not in get_all_collections(client):
            start_time = time.time()
            client.create_collection(collection_name)
            end_time = time.time()
            print(f'Time for creating a new collection for the first time = {end_time - start_time} secs')
            logfile.write(f'Time for creating a new collection for the first time = {end_time - start_time} secs\n')
            st.toast('Created new collection!')
        
        start_time = time.time()
        client.upload_points(collection_name, emb, meta_emb, chunks)
        end_time = time.time()
        print(f'Time for uploading all the points in the collection = {end_time - start_time} secs')
        logfile.write(f'Time for uploading all the points in the collection = {end_time - start_time} secs\n')
        logfile.close()
        
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
    multi_selector(docs_source)
    upload_docs(docs_source)
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
    
    # try:
    #     page1 = requests.get(ap)
    # except requests.exceptions.ConnectionError:
    #     r.status_code = "Connection refused"

    output = requests.post(url='http://127.0.0.1:8000/chat', data=data).text
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