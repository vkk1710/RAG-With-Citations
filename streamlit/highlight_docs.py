import re
import os
import fitz
import pandas as pd

def locate_text(context, text):
    # check - 1
    loc = context.find(text)
    if loc >= 0:
        return loc
    
    #check - 2
    text_set = set(text.split())
    context_set = set(context.split())
    
    if text_set <= context_set:
        return 1
    
    # check - 3
    else:
        absent_tokens = len([tok for tok in text_set if tok not in context_set])
        if absent_tokens/len(text_set) <= 0.2:
            return 1
    
    return -1
    

def iterate_contexts(context, text):
    for i in range(len(context)):
        res = locate_text(context[i], text)
        if res > -1:
            break
    
    if res < 0:
        return -1
    
    return i


def validate_output(context, output):
    context = [re.sub('\n', ' ', c) for c in context]
    context = [c.strip().lower() for c in context]
    context = [' '.join(c.split()) for c in context]
    
    output = output['citations']
    
    final_idxs = []
    
    for out in output:
        idx = out['source_id']-1
        text = out['quote'].strip().lower()
        text = ' '.join(text.split())
        if idx >= len(context):
            res = -1
        else:
            res = locate_text(context[idx], text)
            
        
        # print('LLM output: \n',out)
       
        if res < 0:
            final_idx = iterate_contexts(context, text)
            
            if final_idx < 0:
                print(f"LLM Output is not matching any item in the Context List for citation with source_id: {idx+1}")
                
            else:
                final_idxs.append(final_idx)
        
        else:
            final_idxs.append(idx)
            
        # print('\n\ncontext id: \n', final_idxs, '\n\n')
        
    final_idxs = list(set(final_idxs))
    return final_idxs
            
            
def get_citations(llm_citation_indices, ranked_result):
    return [ranked_result[i] for i in range(len(ranked_result)) if i in llm_citation_indices]

def highlight_text_cell(s, rows):
    to_highlight = ['color: yellow;' if i in rows else '' for i in range(len(list(s)))]
    return to_highlight


def highlight_csv(highlights_list, output_path = './streamlit_documents/highlighted_docs', source_path = './streamlit_documents'):
    docs = list(set([doc['file_name'] for doc in highlights_list]))
    
    highlighted_csvs = []

    for doc in docs:
        file_path = os.path.join(source_path, doc)
        output_file_extension = "_highlighted.xlsx"
        output_file_name = doc.replace(".csv",output_file_extension) 
        output_file_path = os.path.join(output_path, output_file_name)

        highlight_doc = pd.read_csv(file_path)
        
        rows = list(set([d['page'] for d in highlights_list if d['file_name'] == doc]))
        
        # print('rows: ', rows)
            
        highlight_doc = highlight_doc.style.apply(highlight_text_cell, rows = rows, axis=0)
        highlight_doc.to_excel(output_file_path)
        
        highlighted_csvs.append((output_file_name, rows))
        
    return highlighted_csvs
            
               
def highlight_pdf(highlights_list, output_path = './streamlit_documents/highlighted_docs', source_path = './streamlit_documents'):
    
    docs = list(set([doc['file_name'] for doc in highlights_list]))
    
    highlighted_pdfs = []
    
    for doc in docs:
        pages = []
        pdf_path = os.path.join(source_path, doc)
        output_file_extension = "_highlighted.pdf"
        output_file_name = doc.replace(".pdf",output_file_extension) 
        output_pdf_path = os.path.join(output_path, output_file_name)

        highlight_doc = fitz.open(pdf_path)
        
        for doc_dict in highlights_list:
            file = doc_dict['file_name']
            if file != doc:
                continue
            page_num = doc_dict['page']
            text_to_highlight = doc_dict['text']
            
            pages.append(page_num)
            
            page = highlight_doc.load_page(page_num-1)
            text_instances = page.search_for(text_to_highlight.strip())
            
            print('\n\ndoc: \n', re.sub('\n', ' ', text_to_highlight), '\n\n')
            
            print('\npage num: \n', page_num-1, '\n\n')
            
            print('\npage: \n', page, '\n\n')
            
            print('\ntext_instances: \n', text_instances, '\n\n')
            
            for inst in text_instances:
                print("HIGHLIGHTING", inst)
                page.add_highlight_annot(inst)
                
            highlight_doc.save(output_pdf_path, garbage=0, deflate=False, clean=False)
        
        pages = list(set(pages))    
        highlighted_pdfs.append((output_file_name, pages))
    
    return highlighted_pdfs