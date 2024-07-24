import re
import os
import fitz

def iterate_contexts(context, text):
    for i in range(len(context)):
        res = context[i].find(text) 
        if res > -1:
            break
    
    if res < 0:
        return -1
    
    return i


def validate_output(context, output):
    context = [re.sub('\n', ' ', c) for c in context]
    context = [c.lower() for c in context]
    output = output['citations']
    
    final_idxs = []
    
    for out in output:
        idx = out['source_id']-1
        text = out['quote'].lower()
        res = context[idx].find(text) 
        
        # print('LLM output: \n',out)
       
        if res < 0:
            final_idx = iterate_contexts(context, text)
            
            if final_idx < 0:
                print("LLM Output is not matching any item in the Context List.")
                
            else:
                final_idxs.append(final_idx)
        
        else:
            final_idxs.append(idx)
            
        # print('\n\ncontext id: \n', final_idxs, '\n\n')
        
    final_idxs = list(set(final_idxs))
    return final_idxs
            
            
def get_citations(llm_citation_indices, ranked_result):
    return [ranked_result[i] for i in range(len(ranked_result)) if i in llm_citation_indices]

               
def highlight_pdf(highlights_list, output_path = './documents/highlighted_docs', source_path = './documents'):
    
    for doc in highlights_list:
        file = doc['metadata']['file_name']
        page_num = doc['metadata']['page']
        text_to_highlight = doc['text']
        
        pdf_path = os.path.join(source_path, file)
        output_file_extension = "_highlighted.pdf"
        output_file_name = file.replace(".pdf",output_file_extension) 
        output_pdf_path = os.path.join(output_path, output_file_name)

        highlight_doc = fitz.open(pdf_path)
        
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