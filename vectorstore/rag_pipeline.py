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

def run_rag_pipeline():
    connections.connect(host=os.environ.get("MILVUS_HOST"), port=os.environ.get("MILVUS_PORT"))

    prompt = PromptTemplate.from_template(
    """
    Human: You are an AI assistant helping evaluate candidate profiles based on a job requirement.

    Given the following candidate profiles:
    {profiles}

    And the following job requirement or query:
    {query}

    Your tasks:
    1. First, check if the job requirement or query is clear and specific. If the query appears to be a placeholder, typo, or is too vague , do not proceed.
    2. If the query is unclear, respond with: 
       "La description de poste ou la requête fournie n'est pas suffisamment claire pour évaluer la pertinence des candidats. Veuillez reformuler ou fournir une description plus détaillée du poste."
    3. If the query is clear:
       - Rank the candidates from most to least relevant to the job.
       - Assign a relevance score to each candidate from 0 (not relevant) to 10 (highly relevant).
       - Only include candidates with a score greater than 4 in your  response.
       - Use specific details and entity names to support your reasoning.
       - just include the relevant candidates in the response.

    Return the final answer clearly and concisely, in **French**..
    Assistant:
    """
    )
    #llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key=os.environ["MISTRAL_API_KEY"])   

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
    retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3,
        "search_type": "hybrid",
        "alpha": 0.5  
    }
)
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
 