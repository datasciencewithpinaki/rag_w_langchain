from langchain_core.documents.base import Document  # required to add additional metadata
from langchain_community.vectorstores.chroma import Chroma


class RetrieveDocs:

    def main(query:str, vector_store:Chroma, method_search:str='mmr', method_rerank:str='simple', **kwargs)->list[Document]:
        '''
        Retrieve documents from vector store based on given query
        Arguments:
            query <str>: user query
            vector_store <Chroma>: chroma vector database,
            method <str>: choose one out of ['mmr', 'siml', 'siml_w_relvscore', 'siml_w_score'],
            kwargs <dict>: includes a filter based on metadata
        '''
        docs_retrieved = RetrieveDocs.retrieve_docs(query, vector_store, method_search, **kwargs)
        if method_search in ['mmr']:  # re-ranking is inherent in these            
            docs_reranked = docs_retrieved[:]
            print(f'No re-ranking required. `{method_search}` already incorporates re-ranking.')
        else:  # explicit re-ranking is required
            docs_reranked = RetrieveDocs.rerank_docs(query, docs_retrieved, method_rerank)
        return docs_reranked

    def retrieve_docs(query:str, vector_store:Chroma, method:str='mmr', **kwargs)->list[Document]:
        '''
        Retrieve documents from vector store based on given query
        Arguments:
            query <str>: user query
            vector_store <Chroma>: chroma vector database,
            method <str>: choose one out of ['mmr', 'siml', 'siml_w_relvscore', 'siml_w_score'],
            kwargs <dict>: includes a filter based on metadata
        '''
        # parameters
        k = kwargs.get('k')
        k = 4 if not k else k
        metadata_filter = kwargs.get('metadata')
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

    def rerank_docs(query:str, docs:list[Document], method:str='simple')->list[Document]:
        '''
        Rerank documents based on given query and chosen strategy
        Arguments:
            query <str>: user query
            method <str>: choose one out of ['simple', 'multi-model'],
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