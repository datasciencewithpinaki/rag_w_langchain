# Steps to On-board

Let's get started with onboarding onto the new Langchain based RAG pipeline that is under development. 
<br>The idea is to ‘own’ a data source individually and experiment on it. 
 
## Retrieval Chain (for each data source):
	1. Know the data we own
	2. Categorize data source into topics that it can answer from
	3. Clean data as required
		a. QnA vs Plain text document
		b. Tables - how to load them
		c. Images (ignore for now)
	4. Develop a chunking strategy. One of:
		a. Small chunks 
		b. Large chunks 
		c. Both (Parent and Child) 
        d. Markdown separator based
	5. Embedding strategy
		a. Sentence Transformer (for dev)
		b. OpenAI embedding (test in future - not available for WMT?)
	6. Vector DB
		a. Chroma (for dev)
		b. Milvus (test for future)
	7. Retrieval Process
		a. Base retriever ('mmr')
		b. Other search algos like similarity, relevance, etc.
	8. Use Re-ranking 
		a. Without (Pass)
		b. Simple reordering
		c. Cross encoder, flashrank, etc. (test for future)
	9. Evaluate retrieved documents based on user query
	10. Save retrieval evaluation metrics to a file
		a. Maintain the history
		b. Monitor metrics

## Generation Chain:
	1. LLM model
		a. OpenAI (in dev)
		b. Llama2 (test)
	2. Experiments with Prompt
		a. Simple text based prompt
		b. Using PromptTemplate and ChatPromptTemplate, Message Types (Human, User, AI, System), etc.
	3. Evaluate generated output based on user query
		a. Monitor metrics
	4. Save generation evaluation metrics to a file
		a. Maintain the history
		b. Monitor metrics


## LLMOps & Eng:
	1. Deploy to Dev
	2. Load testing
	3. UAT 
	4. Release


