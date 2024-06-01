# Things to Try in LLM & Langchain

## Start with a single document

- [x] Test AzureOpenAI `LLM` through WMT gateway
- [x] Create a simple document for RAG
- [x] Using Langchain, `Load` & `Split` document to create `chunks`
    - Test multiple strategies based on: `chunk_size` & `chunk_overlap`
- [x] Create `Embedding` for document 
    - Use AzureOpenAI Embedding through WMT gateway
    - Use other opensource Embedding 
- [x] Create a Vector DB
- [x] Create a Retriever
    - Test multiple strategies based on `k`, `retrieval_strategy`, `similarity_metric`, `threhsold`
- [x] Create a full LLM chain
- [x] Pass multiple queries to test output

## Repeat with multiple documents

- [x] Create multiple plain text documents for RAG
- [x] Using Langchain, `Load` & `Split` document to create `chunks`
    - Test multiple strategies based on: `chunk_size` & `chunk_overlap`
- [x] Create `Embedding` for document 
    - Use AzureOpenAI Embedding through WMT gateway
    - Use other opensource Embedding 
- [x] Create a Vector DB
- [x] Create a Retriever
    - Test multiple strategies based on `k`, `retrieval_strategy`, `similarity_metric`, `threhsold`
- [x] Create a full LLM chain
- [x] Pass multiple queries to test output

## Add metadata to each document

- [x] Add metadata to each document [Link >> 4](#Links)
- [x] Enable filter based on metadata while retrieving [Link >> 5, 6](#Links)

## Log & store model performance improvement for each iteration 

- [ ] step 1
- [ ] step 2
- [ ] step 3


## Other Experimentation Areas

- [ ] Using Tools and Agents (`NotStarted`)
- [ ] Using Parent and Child Splitter (for different chunk_size and overlap_size) (`WIP`)
- [ ] Using re-ranking strategy on retrieved documents (`WIP`)
- [ ] Knowledge Graph + RAG [Link >> 1](#Links) (`NotStarted`)
- [ ] Extracting entity from prompt / query to answer questions [Link >> 3](#Links) (`NotStarted`)


# Error Resolutions
1. OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.
> python3 -m spacy download en_core_web_sm


# Links
1. [Raptor RAG kg enhanced](https://github.com/leannchen86/raptor-rag-kg-enhanced/blob/main/raptor-rag-kg-enhanced.ipynb)

2. [semi-structured rag](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb)

3. [entity relationship extaction](https://github.com/langchain-ai/langchain/blob/master/cookbook/extraction_openai_tools.ipynb)

4. [How to add metadata](https://medium.com/@sandyshah1990/exploring-rag-implementation-with-metadata-filters-llama-index-3c6c08a83428)

5. [Different filter options](https://docs.pinecone.io/guides/data/filter-with-metadata#considerations-for-serverless-indexes)

6. [multiple filters in retriever](https://github.com/langchain-ai/langchain/discussions/10537)

7. [compression reranking (langchain)](https://python.langchain.com/v0.1/docs/integrations/retrievers/flashrank-reranker/)

8. [cross encoder reranking (langchain)](https://python.langchain.com/v0.1/docs/integrations/document_transformers/cross_encoder_reranker/)

9. [Chat history aware retrieval (langchain)](https://www.linkedin.com/pulse/beginners-guide-conversational-retrieval-chain-using-langchain-pxhjc/)
