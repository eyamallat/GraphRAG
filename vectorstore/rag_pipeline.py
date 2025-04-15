import json
import os
from vectorstore.milvus_handler import create_documents 
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from pymilvus import connections
from langchain_mistralai import ChatMistralAI


def run_rag_pipeline():
    connections.connect(host=os.environ.get("MILVUS_HOST"), port=os.environ.get("MILVUS_PORT"))
    
    prompt = PromptTemplate.from_template(
        """
        "Given the job requirement: \"{query}\",
        rank the following candidate profiles from most to least relevant 
        give a score for each relevant candidate from 0 to 10  
        translate the answer to french   
        \n\n   
        {profiles}
        """
    )
    
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key=os.environ["MISTRAL_API_KEY"])   

    with open('./candidates.json', 'r', encoding='utf-8') as f:
        data = json.load(f)  
    doc_splits = create_documents(data)

    vectorstore = Milvus.from_documents(
            documents=doc_splits,
            collection_name="rag_milvus",
            embedding=HuggingFaceEmbeddings(),
            connection_args={
            "host": os.environ.get("MILVUS_HOST"), 
            "port": os.environ.get("MILVUS_PORT"),
        },
            drop_old=True,
    ) 
        
    retriever = vectorstore.as_retriever()
    chain = prompt | llm | StrOutputParser()
    
    try:
        while True:
            
            query = input("\nAsk a question (or type 'exit' to quit): ")
            if query.lower() == "exit":
                break
            
            top_docs = retriever.invoke(query)[:2]
            profiles = "\n\n".join([doc.page_content for doc in top_docs])
            
                
            answer = chain.invoke(
                {"query": query, "profiles": profiles},
            )
            print("\nAnswer:", answer)
        
    
    

        
    finally:
        connections.disconnect()