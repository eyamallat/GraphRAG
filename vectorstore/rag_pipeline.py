import json
import os
from vectorstore.milvus_handler import create_documents 
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from pymilvus import (
    connections,
    utility,
    
)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def create_retriever():
    connections.connect(host=os.environ.get("MILVUS_HOST"), port=os.environ.get("MILVUS_PORT"))

    with open('./candidates.json', 'r', encoding='utf-8') as f:
        data = json.load(f) 
             
    doc_splits = create_documents(data)
    
    analyzer_params_custom = {
    "tokenizer": "standard",
    "filter": [
        "lowercase",
    ],
}
    model=HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct')

    vectorstore = Milvus.from_documents(
        collection_name="rag_milvus",
        documents=doc_splits,
        embedding=model,
        builtin_function=BM25BuiltInFunction(output_field_names="sparse",
                                             analyzer_params=analyzer_params_custom,),
        vector_field=["dense", "sparse"],
        connection_args={
            "host": os.environ.get("MILVUS_HOST"), 
            "port": os.environ.get("MILVUS_PORT"),
        },
        consistency_level="Strong",
        drop_old=True,
    )
    return vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "search_type": "hybrid",
        "alpha": 0.5  
    }
)
    
    
def run_rag_pipeline():
    retriever=create_retriever()
    prompt = PromptTemplate.from_template(
    """
    Human: You are an AI assistant helping evaluate candidate profiles based on a job requirement.

    Given the following candidate profiles:
    {profiles}

    And the following job requirement or query:
    {query}

    Your tasks:

    1. Check if the query or the job requirement is completely meaningless (like random characters or unrelated text). A clear query should at least include the job title and optionally skills, experience, or education. If the query is meaningless , consider it unclear.

    3. If the query is clear:
       - Rank the candidates from most to least relevant to the job.
       - Assign a relevance score to each candidate from 0 (not relevant) to 10 (highly relevant).
       - Only include candidates with a score strictly greater than 4 in your response.
       - Use specific details (e.g., job titles, skills, experiences) to support your ranking.
       - Include only the relevant candidates in the output.

    Return the final answer clearly and concisely, in **French**.
    Assistant:
    """
)

    llm = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key=os.environ["MISTRAL_API_KEY"])       
    
    chain = (
        {
            "query": RunnablePassthrough(),
            "profiles": lambda q: format_docs(retriever.invoke(q)),

        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    try:
        while True:
            query = input("\nAsk a question (or type 'exit' to quit): ")
            if query.lower() == "exit":
                break

            answer = chain.invoke(query)
            print("\nAnswer:", answer)
             
        
    finally:
        connections.disconnect("default")
 