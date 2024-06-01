from typing import List
from langchain_core.documents.base import Document  # required to add additional metadata
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from codes.llm_gateway import openAI_wo_api
from codes.retrieve_docs import ReRanking, EmbeddingSimilarity

class LlmWithManualRag:
    '''
    Create an LLM Chain
    If retriever is None, LLM is invoked without using retrieved documents
    '''

    def invoke_chain(prompt_upd:str):
        '''
        Get response from LLM by passing the prompt. 
        This may include the retrieved documents if `add_context_to_prompt()` method is called earlier.
        '''
        model = openAI_wo_api()
        response = model.invoke(prompt_upd)
        return response
    
    def add_context_to_prompt(query:str, 
                              docs_retrieved:List[Document], 
                              rerank:bool=False, 
                              rerank_method:str='pass'):
        '''
        Add context to the query and this would be the prompt for generation later.
        prompt_upd = context + query
        where context = docs_as_concatenated_txt after reranking as required.
        Arguments:
            query <str>: user query
            docs_retrieved <List[Document]>: list of retrieved docs
            rerank <bool>: True if reranking is required
            rerank_method <str>: choose one out of ['pass', 'simple']; Is ignored if rerank is False
        '''
        model = openAI_wo_api()

        docs_retrieved_upd = docs_retrieved if not rerank else LlmWithManualRag.rerank_docs(docs_retrieved, rerank_method)
        
        docs_retrieved_as_txt = LlmWithManualRag.docs_to_plaintext(docs_retrieved_upd)
        
        prompt_upd = f"Context: {docs_retrieved_as_txt}" + '\n' + f"""
        Answer the question based only on the context provided. 
        If you don't know the answer, say you do not know. 
        Decide based on the question if answer can be made concise or not. 
        If so, keep answer within three sentences. Concise is better.
        If answer needs to be elaborate, generate a very structured response.
        Question: {query}
        """
        return prompt_upd

    
    def docs_to_plaintext(docs_retrieved:List[Document])->str:
        docs_retrieved_as_txt = ';'.join([doc.page_content for doc in docs_retrieved])
        return docs_retrieved_as_txt

    
    def filter_docs_on_siml(query:str, docs:List[Document], thresh:float=0.5, k:int=4)->List[Document]:
        '''
        find cosine similarity between query and the retrieved docs
        filter only for docs where siml > thresh
        pick a max of k docs
        '''
        docs_as_text = [doc.page_content for doc in docs]
        siml_scores = EmbeddingSimilarity.get_similarity([query], docs_as_text)
        siml_scores = siml_scores[0]
        print(siml_scores)
        siml_scores_idx = siml_scores>thresh
        print(siml_scores_idx)
        docs_filtd = [key for key,val in zip(docs, siml_scores) if val>thresh]
        docs_filtd = docs_filtd[0:k]
        return docs_filtd


    def rerank_docs(docs:list[Document], method:str='simple')->list[Document]:
        '''
        Rerank documents based on given query and chosen strategy
        Arguments:
            query <str>: user query
            method <str>: choose one out of ['pass', 'simple'],
        '''
        methods = ['pass', 'simple', 'multi-model']
        if method not in methods:
            print(f'Did not rerank as method is incorrect. Method needs to be out of {methods}')
            docs_reranked = docs[:]
        elif method=='pass':
            docs_reranked = docs[:]
        elif method=='simple':  # Example: [1,2,3,4,5,6] --> [1,3,5] + [6,4,2]
            docs_even = [doc for idx, doc in enumerate(docs) if idx%2==0]
            docs_odd = [doc for idx, doc in enumerate(docs) if idx%2!=0]
            docs_odd_reversed = docs_odd[::-1]
            docs_reranked = docs_even + docs_odd_reversed
        elif method=='multi-model':
            print(f'As of now, `{method}` has not been implemented')
            raise NotImplementedError
        return docs_reranked


class LlmWithRag:
    
    def create_chain(retriever:List[Document], rerank:bool=False, rerank_method:str=None):
        '''
        Create an LLM Chain that includes retrieved documents
        '''
        model = openAI_wo_api()
        
        system_prompt = ("""
        Answer the question based only on the context provided. 
        If you don't know the answer, say you do not know. 
        Decide based on the question if answer can be made concise or not. 
        If so, keep answer within three sentences. Concise is better.
        If answer needs to be elaborate, generate a very structured response.
        Context: {context}
        Question: {input}
        """)
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt
        )

        if rerank:
            retriever_upd = ReRanking.rerank_with(retriever, rerank_method)
            
        elif not rerank:
            retriever_upd = retriever
        
        retrieval_chain = create_retrieval_chain(
            retriever_upd,
            chain
        )

        return retrieval_chain