from langchain.document_loaders import TextLoader, Docx2txtLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document  # required to add additional metadata

from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

class Data2Docs:

    def main(path:str, metadata:dict=None, **kwargs):
        '''
        entry point for data read: file read, add metadata, create chunks
        Arguments:
            path <str>: path to individual file
            metadata <dict>: dict having key value pairs of metadata for this file
            **kwarfs <dict>: dict that has key value pairs for chunking strategy
        '''
        docs = Data2Docs.load_document(path)
        docs_w_md = [Data2Docs.add_metadata_to_doc(doc, metadata) for doc in docs]
        doc_chunks = Data2Docs.split_document_window(docs_w_md, **kwargs)
        return doc_chunks
    
    def load_document(path:str):
        '''
        load document from path
        '''
        file_ext = Data2Docs.check_file_type(path)
        if file_ext in ['txt', 'csv']:
            loader = TextLoader(path)
        elif file_ext in ['docx']:
            loader = Docx2txtLoader(path)
        elif file_ext in ['pdf']:
            loader = PyPDFLoader(path)
        else:
            print('file type needs to be from either of [txt, csv, docx]')
            raise NotImplementedError
        docs = loader.load()
        return docs

    def add_metadata_to_doc(doc:Document, metadata:dict=None)->Document:
        '''
        Add or update metadata to existing Document
        '''
        if not metadata:
            return doc
        page_content = doc.page_content
        metadata_upd = doc.metadata
        metadata_upd.update(metadata)
        doc_w_metadata = Document(page_content=page_content, metadata=metadata_upd)
        return doc_w_metadata
    
    def split_document_window(docs:list[Document], **kwargs):
        '''
        split document based on given chunk size & overlap
        Arguments
            docs <list[Document]>: list of docs loaded from file
            **kwarfs <dict>: dict that has key value pairs for chunking strategy
        '''
        chunk_size = kwargs.get('chunk_size')
        chunk_size = 200 if not chunk_size else chunk_size
        chunk_overlap = kwargs.get('chunk_overlap')
        chunk_overlap = 10 if not chunk_overlap else chunk_overlap
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        splitDocs = splitter.split_documents(docs)
        return splitDocs

    # helper funcs
    def check_file_type(file:str):
        '''
        detect file extension
        '''
        file_ext = file.split('.')[-1]
        return file_ext
    

class Docs2VectorDb:

    COLLECTION_NAME = 'RB_KB_DOCS'

    def main(docs:list[Document], persist_directory:str, model_name:str=None)->Chroma:
        '''
        create a vector store with these initial documents; persist this db
        '''
        vector_store = Docs2VectorDb.create_vector_store(docs, persist_directory, model_name)
        return vector_store

    def get_embedding_func(model_name:str=None):
        '''
        Embed document based on chosen model
        '''
        model_name = "all-MiniLM-L6-v2" if not model_name else model_name
        embedder = SentenceTransformerEmbeddings(
            model_name=model_name, 
        )
        return embedder

    def create_vector_store(docs:list[Document], persist_directory:str, model_name:str=None)->Chroma:
        '''
        create a vector store with these initial documents; persist this db
        '''
        vector_store = Chroma.from_documents(
            docs, 
            Docs2VectorDb.get_embedding_func(model_name), 
            # metadatas=[{'topics': 'marketing, sales'}],
            collection_name=Docs2VectorDb.COLLECTION_NAME,
            # collection_metadata={'key1': 'val1'}
            persist_directory=persist_directory
            )
        return vector_store

    def add_docs_to_vector_db(vector_store:Chroma, new_docs:list[Document])->Chroma:
        '''
        add more documents to an existing vector db
        '''
        _ = vector_store.add_documents(
            new_docs,
            # metadatas=[{'topics': 'finance, money'}],
            )
        print(f'added {len(new_docs)} indices to vector store')
        print(f'sources available after insertion:\n{Docs2VectorDb.sources_from_vdb(vector_store)}')
        return vector_store

    def delete_docs_using_metadata(vector_store:Chroma, metadata_filter:dict)->Chroma:
        '''
        delete existing documents from a vector db
        '''
        docs_filtd = vector_store.get(where=metadata_filter, include=['metadatas'])
        docs_filtd_id = docs_filtd['ids']
        _ = vector_store.delete(docs_filtd_id)
        print(f'deleted {len(docs_filtd_id)} indices from vector store')
        print(f'sources available after deletion:\n{Docs2VectorDb.sources_from_vdb(vector_store)}')
        return vector_store

    def update_docs_using_metadata(vector_store:Chroma, 
                                   new_docs:list[Document], 
                                   metadata_filter:dict)->Chroma:
        '''
        step1: delete existing documents from a vector db
        step2: add documents to the vector db
        '''
        vector_store = Docs2VectorDb.delete_docs_using_metadata(vector_store, metadata_filter)
        vector_store = Docs2VectorDb.add_docs_to_vector_db(vector_store, new_docs)
        return vector_store

    def load_vector_store(path_persist_db:str):
        '''
        load vector store from persistent path to memory
        '''
        vector_store = Chroma(
            collection_name=Docs2VectorDb.COLLECTION_NAME,
            persist_directory=path_persist_db, 
            embedding_function=Docs2VectorDb.get_embedding_func()
        )
        return vector_store

    def sources_from_vdb(vector_store:Chroma):
        docs = vector_store.get()['metadatas']
        metadata_sources = {}
        metadata_sources['source'] = []
        for doc in docs:
            for k,v in doc.items():
                if k!='source':
                    continue
                metadata_sources[k].append(v)
        metadata_sources['source'] = {v for v in metadata_sources['source']}
        return metadata_sources
        

