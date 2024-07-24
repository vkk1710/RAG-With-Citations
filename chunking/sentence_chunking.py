from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from typing import List
from llama_index.core.ingestion import run_transformations

class SentenceChunker:
    def __init__(self, file_dir:List[str]):
        reader = SimpleDirectoryReader(input_dir=file_dir)
        self.docs = reader.load_data()
        self.splitter = SentenceSplitter.from_defaults(
                                            chunk_size=128,
                                            chunk_overlap=32,
                                            separator=' ', 
                                            paragraph_separator='\n\n\n', 
                                            secondary_chunking_regex='[^,.;。？！]+[,.;。？！]?'
                                        )
    
    def chunk(self):
        nodes = run_transformations(
                self.docs,  # type: ignore
                transformations=[self.splitter],
            )
        return [node.text for node in nodes]
