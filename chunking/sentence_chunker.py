# Author - Kamal

from typing import List
import fitz
import os
import spacy
import time
import re
import pandas as pd


def check_tsb(doc, header_y = 100):
    num_pages = 1
    start_page = 0
    
    if len(doc) >= 3:
        num_pages = 3
        
    for pg in range(num_pages):
        page = doc.load_page(pg)
        rect = page.rect
        clip = fitz.Rect(0, 0, rect.width, header_y)
        header = page.get_text(clip=clip)
        
        match = re.search(r'[TECHNICAL SERVICE BULLETIN|TSB ?[0-9-]{2,7}', header)
        
        if match != None: 
            start_page = pg
            return True, start_page 
    
    return False, start_page


def extract_tsb_metadata(doc, start_page):
    metadata = ''
    
    start_page_text = doc.load_page(start_page).get_text()
             
    # match = re.search(r'Parts|SERVICE PROCEDURE|WARRANTY STATUS|Warranty Status:', start_page_text)
    # match = re.search(r'((Action|ACTION).+)\n(Parts|SERVICE PROCEDURE|WARRANTY STATUS|Warranty Status:)', start_page_text)
    
    act_match = re.search(r'Action:|ACTION', start_page_text)

    if act_match != None:
        start = act_match.start()
        metadata += start_page_text[:start]
        act_text = start_page_text[start:]
        
        match = re.search(r'\n(Parts|SERVICE PROCEDURE|WARRANTY STATUS|Warranty Status:)', act_text)
        
        if match != None:
            print(doc)
            print(start_page)
            print(match)
            start = match.start()
            metadata += act_text[:start]
            
        else:
            print(f'Unable to extract Metadata for {doc} page num = {start_page} due to none of the sections match!!')
            
    else:
        print(f'Unable to extract Metadata for {doc} page num = {start_page} due to "Action" keyword!!')
    
    print(metadata, '*'*30, '\n')
        
    return metadata

def pdf_reader(inp_dir, document_names):
    docs_content = []
    # docs_list = [doc for doc in os.listdir(inp_dir) if doc in document_names]
    docs_list = os.listdir(inp_dir)
    for doc_name in docs_list:
        if (doc_name.split('.')[-1] != "pdf"):
            continue
        doc_path = os.path.join(inp_dir, doc_name)
        doc = fitz.open(doc_path)

        # Extract text and metadata from each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            page_content = {
                "text": text,
                "page": page_num+1,
                "file_name": doc_name
            }
            docs_content.append(page_content)
            
    return docs_content


def csv_reader(inp_dir, document_names):
    docs_content = []
    # docs_list = [doc for doc in os.listdir(inp_dir) if doc in document_names]
    docs_list = os.listdir(inp_dir)
    for doc_name in docs_list:
        if (doc_name.split('.')[-1] == "csv"):
            file_path = os.path.join(inp_dir, doc_name)
            data = pd.read_csv(file_path).CDESCR.to_list()
            
            doc_dict = {
                "file_name": doc_name,
                "data": data
            }
            
            docs_content.append(doc_dict)
        
    return docs_content

# Currently it just extracts text if its a tsb. If not, then it just ignores the pdf. Change it!
def tsb_reader(inp_dir):
    docs_content = []
    docs_list = os.listdir(inp_dir)
    
    for doc_name in docs_list:
        if (doc_name.split('.')[-1] != "pdf"):
            continue
        doc_path = os.path.join(inp_dir, doc_name)
        doc = fitz.open(doc_path)
        
        # check if the pdf is tsb or not
        
        is_tsb, start_page = check_tsb(doc)
        
        if not is_tsb:
            continue
        
        metadata = extract_tsb_metadata(doc, start_page)
        metadata = re.sub('\n', ' ', metadata)

        # Extract text and metadata from each page
        for page_num in range(start_page, len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            page_content = {
                "metadata": metadata,
                "text": text,
                "page": page_num+1,
                "file_name": doc_name
            }
            docs_content.append(page_content)
            
    return docs_content


class PDFSentenceChunker:
    def __init__(self, file_dir, document_names):
        start_time = time.time()    
        self.docs = pdf_reader(inp_dir=file_dir, document_names=document_names)
        end_time = time.time()
        print(f'Time for loading pdfs from pdf_reader = {end_time - start_time} secs')    
        
        start_time = time.time()    
        self.splitter = spacy.load("en_core_web_sm")
        # self.splitter.disable_pipe("parser")
        # self.splitter.enable_pipe("senter")
        end_time = time.time()
        print(f'Time for loading the spacy model for sentence splitting = {end_time - start_time} secs')  
    
    def chunk(self):
        docs_content = []
        max_length = 64
        
        print(f'No of doc_dicts = {len(self.docs)}\n')
        for doc_dict in self.docs:
            text = doc_dict['text']
            pg_num = doc_dict['page']
            file_name = doc_dict['file_name']
            
            # start_time = time.time()
            doc = self.splitter(text)
            sentences = [sentence.text for sentence in doc.sents]
            # end_time = time.time()
            # print(f'Time for splitting sentences = {end_time - start_time} secs')   
            
            # start_time = time.time()
            chunks = []
            chunk = ''
            for sent in sentences:
                if len(chunk.split()) + len(sent.split()) <= max_length:
                    chunk += ' ' + sent
                else:
                    chunks.append(chunk)
                    chunk = sent
            
            if chunk != sent:
                chunks.append(chunk)
            
            content = [{
                "text": chunk,
                "page": pg_num,
                "file_name": file_name
                } for chunk in chunks
            ]
            docs_content += content 
            # end_time = time.time()
            # print(f'Time for arranging the splitted sentences into chunks = {end_time - start_time} secs\n')  
            
        print(f'No of docs_content after chunking = {len(docs_content)}')
        return docs_content
    
    
    def split_sentences(self, doc_dict):
        text = doc_dict['text']
        pg_num = doc_dict['page']
        file_name = doc_dict['file_name']
        
        doc = self.splitter(text)
        sentences = [sentence.text for sentence in doc.sents]
        
        return (sentences, pg_num, file_name)

    
    def chunk_new(self):
        docs_content = []
        max_length = 64
        
        print(f'No of doc_dicts = {len(self.docs)}\n')
        
        start_time = time.time()
        
        # doc_tups = map(self.split_sentences, self.docs) 
        
        # doc_tups = [([sent.text for sent in self.splitter(doc['text']).sents], doc['page'], doc['file_name']) for doc in self.docs]
        
        doc_tups = []
             
        texts = [d['text'] for d in self.docs]
        pg_nums = [d['page'] for d in self.docs]
        fns = [d['file_name'] for d in self.docs]   
          
        for i, doc in enumerate(self.splitter.pipe(texts, n_process=-1)):
            sentences = [sentence.text for sentence in doc.sents]
            doc_tups.append((sentences, pg_nums[i], fns[i]))
        end_time = time.time()
        print('Output of splitting sentences using map - ', list(doc_tups)[:2])
        print(f'Time for splitting sentences using map = {end_time - start_time} secs')   
        
        start_time = time.time()
        for doc in doc_tups:
            chunks = []
            chunk = ''
            
            sentences = doc[0]
            pg_num = doc[1]
            file_name = doc[2]
            
            for sent in sentences:
                if len(chunk.split()) + len(sent.split()) <= max_length:
                    chunk += ' ' + sent
                else:
                    chunks.append(chunk)
                    chunk = sent
            
            if chunk != sent:
                chunks.append(chunk)
            
            content = [{
                "text": chunk,
                "page": pg_num,
                "file_name": file_name
                } for chunk in chunks
            ]
            docs_content += content 
        
        end_time = time.time()
        print(f'Time for arranging the splitted sentences into chunks = {end_time - start_time} secs\n') 
        
        print(f'No of docs_content after chunking = {len(docs_content)}') 
            
        return docs_content


class TSBSentenceChunker:
    def __init__(self, file_dir, document_names):
        start_time = time.time()    
        self.docs = tsb_reader(inp_dir=file_dir)
        end_time = time.time()
        print(f'Time for loading pdfs from pdf_reader = {end_time - start_time} secs')    
        
        start_time = time.time()    
        self.splitter = spacy.load("en_core_web_sm")
        # self.splitter.disable_pipe("parser")
        # self.splitter.enable_pipe("senter")
        end_time = time.time()
        print(f'Time for loading the spacy model for sentence splitting = {end_time - start_time} secs')  
    
    def chunk(self):
        docs_content = []
        max_length = 64
        
        print(f'No of doc_dicts = {len(self.docs)}\n')
        for doc_dict in self.docs:
            metadata = doc_dict['metadata']
            text = doc_dict['text']
            pg_num = doc_dict['page']
            file_name = doc_dict['file_name']
            
            # start_time = time.time()
            doc = self.splitter(text)
            sentences = [sentence.text for sentence in doc.sents]
            # end_time = time.time()
            # print(f'Time for splitting sentences = {end_time - start_time} secs')   
            
            # start_time = time.time()
            chunks = []
            chunk = ''
            for sent in sentences:
                if len(chunk.split()) + len(sent.split()) <= max_length:
                    chunk += ' ' + sent
                else:
                    chunks.append(chunk)
                    chunk = sent
            
            if chunk != sent:
                chunks.append(chunk)
            
            content = [{
                "metadata": metadata,
                "text": chunk,
                "page": pg_num,
                "file_name": file_name
                } for chunk in chunks
            ]
            docs_content += content 
            # end_time = time.time()
            # print(f'Time for arranging the splitted sentences into chunks = {end_time - start_time} secs\n')  
            
        print(f'No of docs_content after chunking = {len(docs_content)}')
        return docs_content
        

class CSVDataLoader:
    def __init__(self, file_dir, document_names):
        self.docs = csv_reader(file_dir, document_names)
    
    def load_data(self):
        docs_content = []
        
        for doc in self.docs:
            file = doc['file_name']
            data_list = doc['data']
            
            content = [{
                    "text": text,
                    "page": idx,
                    "file_name": file
                } for idx, text in enumerate(data_list)
            ]
            docs_content += content
            
        return docs_content