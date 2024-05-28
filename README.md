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

- [x] Add metadata to each document ([How to add metadata](https://medium.com/@sandyshah1990/exploring-rag-implementation-with-metadata-filters-llama-index-3c6c08a83428))
- [x] Enable filter based on metadata while retrieving ([Different filter options](https://docs.pinecone.io/guides/data/filter-with-metadata#considerations-for-serverless-indexes))

## Log & store model performance improvement for each iteration 

- [ ] step 1
- [ ] step 2
- [ ] step 3


## Other Experimentation Areas

- Meta data based filtering during retrieval
- Using Tools and Agents
- Using Parent and Child Splitter (for different chunk_size and overlap_size)
- Using re-ranking strategy on retrieved documents
- Knowledge Graph + RAG 


# Error Resolutions
1. OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.
> python3 -m spacy download en_core_web_sm


# Links
[Raptor RAG kg enhanced](https://github.com/leannchen86/raptor-rag-kg-enhanced/blob/main/raptor-rag-kg-enhanced.ipynb)
