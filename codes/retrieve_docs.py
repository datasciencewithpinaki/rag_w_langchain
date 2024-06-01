from langchain_core.documents.base import Document  # required to add additional metadata
from langchain_community.vectorstores.chroma import Chroma


class RetrieveDocs:

    def main(query:str, 
             vector_store:Chroma, 
             method_search:str='mmr', 
             rerank:bool=False, 
             method_rerank:str='simple', **kwargs)->list[Document]:
        '''
        Retrieve documents from vector store based on given query
        Arguments:
            query <str>: user query
            vector_store <Chroma>: chroma vector database,
            method_search <str>: choose one out of ['mmr', 'siml', 'siml_w_relvscore', 'siml_w_score'],
            method_rerank <str>: choose one out of ['pass', 'simple', 'multi-model'],
            kwargs <dict>: includes a filter based on metadata. This may include:
                k: number of docs to be retrieved
                metadata: for filtering docs based on metadata (Refer vector store to find keys in metadata)
        '''
        docs_retrieved = RetrieveDocs.retrieve_docs(query, vector_store, method_search, **kwargs)
        if not rerank: # no reranking required
            docs_reranked = docs_retrieved[:]
            return docs_reranked
        if method_search in ['mmr']:  # re-ranking inherent in these
            docs_reranked = docs_retrieved[:]
            print(f'No re-ranking required. `{method_search}` already incorporates re-ranking')
        else:
            print("docs_retrieved:\n")
            RetrieveDocs.pprint_docs(docs_retrieved)
            docs_reranked = RetrieveDocs.rerank_docs(query, docs_retrieved, method_rerank)
        return docs_reranked

    def retrieve_docs(query:str, vector_store:Chroma, method:str='mmr', **kwargs)->list[Document]:
        '''
        Retrieve documents from vector store based on given query
        Arguments:
            query <str>: user query
            vector_store <Chroma>: chroma vector database,
            method <str>: choose one out of ['mmr', 'siml', 'siml_w_relvscore', 'siml_w_score'],
            kwargs <dict>: includes a filter based on metadata. This may include:
                k: number of docs to be retrieved
                metadata: for filtering docs based on metadata (Refer vector store to find keys in metadata)
        '''
        # parameters
        k = kwargs.get('k')
        k = 4 if not k else k
        metadata_filter = kwargs.get('filter')
        metadata_filter = {} if not metadata_filter else metadata_filter
        methods = ['mmr', 'siml', 'siml_w_relvscore', 'siml_w_score']
        # criteria
        if method=='mmr':
            docs_retrieved = vector_store.max_marginal_relevance_search(query, k=k, filter=metadata_filter)
        elif method=='siml':
            docs_retrieved = vector_store.similarity_search(query, k=k, filter=metadata_filter)
        elif method=='siml_w_relvscore':
            docs_retrieved = vector_store.similarity_search_with_relevance_scores(query, k=k, filter=metadata_filter)
        elif method=='siml_w_score':
            docs_retrieved = vector_store.similarity_search_with_score(query, k=k, filter=metadata_filter)
        else:
            print(f'method is incorrect. method needs to be out of {methods}')
            raise NotImplementedError

        return docs_retrieved

    def rerank_docs(query:str, docs:list[Document], method:str='simple', **kwargs)->list[Document]:
        '''
        Rerank documents based on given query and chosen strategy
        Arguments:
            query <str>: user query
            method <str>: choose one out of ['pass', 'simple', 'multi-model'],
        '''
        methods = ['pass', 'simple', 'multi-model']
        if method not in methods:
            print(f'method is incorrect. method needs to be out of {methods}')
            docs_reranked = docs[:]
        elif method=='pass':
            docs_reranked = docs[:]
        elif method=='simple':  # Example: [1,2,3,4,5,6] --> [1,3,5] + [6,4,2]
            docs_even = [doc for idx, doc in enumerate(docs) if idx%2==0]
            docs_odd = [doc for idx, doc in enumerate(docs) if idx%2!=0]
            docs_odd_reversed = docs_odd[::-1]
            docs_reranked = docs_even + docs_odd_reversed
        return docs_reranked


    def pprint_docs(docs:list[Document])->None:
        '''
        print docs one by one
        '''
        print("-"*30)
        for doc in docs:
            print(doc.page_content)
            metadata_to_be_printed = {k:v for k,v in doc.metadata.items() if k in ['data_type', 'topic']}
            print(metadata_to_be_printed)
            print('\n')


import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingSimilarity:

    def get_embedding(l_texts:list, model_name=None):
        '''
        create an embedding for each of the elements in a str
        default model: SentenceTransformer("all-MiniLM-L6-v2")
        '''
        print("Starting to Embed texts ...")
        model_name = 'all-MiniLM-L6-v2' if not model_name else model_name
        model_embd = SentenceTransformer(model_name)
        # embeddings
        embds = model_embd.encode(l_texts)
        return embds

    def get_similarity(list1, list2, sim_model_name:str='cosine')->np.matrix:
        '''
        pair wise (cosine) similarity between each of the elements in the two lists
        '''
        # docs_as_text = [doc.page_content for doc in docs]
        if sim_model_name=='cosine':
            list1_embds = EmbeddingSimilarity.get_embedding(list1)
            list2_embds = EmbeddingSimilarity.get_embedding(list2)
            sim_pairwise = cosine_similarity(list1_embds, list2_embds)
        else: # not implemented
            sim_pairwise = None
            raise NotImplementedError
        return sim_pairwise


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.document_compressors import JinaRerank
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

class ReRanking:

    def rerank_with(retriever:List[Document], method:str)->List[Document]:
        '''
        update retriever to include reranking based on the reranking method provided
        '''
        methods = ['flash']
        
        if not method:  # method is None
            retriever_upd = retriever
            
        elif method=='flash':
            retriever_upd = ReRanking.flash_reranker(retriever)

        elif method=='jina':
            retriever_upd = ReRanking.jina_reranker(retriever)

        elif method=='hf_crossencoder':
            retriever_upd = ReRanking.hf_crossencoder_reranker(retriever)
            
        elif method not in methods:
            raise NotImplementedError
        
        return retriever_upd
    

    def flash_reranker(retriever:List[Document])->List[Document]:
        compressor = FlashrankRerank()
        retriever_upd = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
            )
        return retriever_upd

    def jina_reranker(retriever:List[Document])->List[Document]:
        compressor = JinaRerank()
        retriever_upd = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
            )
        return retriever_upd

    def hf_crossencoder_reranker(retriever:List[Document])->List[Document]:
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=model, top_n=3)
        retriever_upd = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
            )
        return retriever_upd