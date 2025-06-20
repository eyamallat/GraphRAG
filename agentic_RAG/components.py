from langchain.tools.retriever import create_retriever_tool
from langchain_mistralai import ChatMistralAI
import os
import json
from agentic_RAG.utils import create_documents
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import (
    connections,
    utility
)

class State(TypedDict):
        thread_id: str
        messages: Annotated[list[AnyMessage], add_messages]

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

retriever=create_retriever()

retrieve= create_retriever_tool(
    retriever,
    "retrieve_candidates",
    "Ranks and scores candidates based on job requirements",
)

GENERATE_PROMPT = PromptTemplate.from_template(
        """
    Human: You are an AI assistant helping evaluate candidate profiles based on a job requirement.

    Given the following candidate profiles:
    {profiles}

    And the following job requirement or query:
    {query}

    Your tasks:
        - Rank the candidates from most to least relevant to the job.
        - Assign a relevance score to each candidate from 0 (not relevant) to 10 (highly relevant).
        - Only include candidates's ids with a score strictly greater than 4 in your answer.
        - Use specific details (e.g., job titles, skills, experiences) to support your ranking.
        - Don't show the irrelevant candidates in the answer.
        -Translate the answer to french.

    Return the answer clearly and concisely, in **French**.
    Assistant:

    """
)

llm = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key=os.environ["MISTRAL_API_KEY"],tools=[])  


def generate_answer(state: State):
    """Generate an answer."""
    question = state["messages"][0].content
    print("---GENERATE---")
    
    chain = (
        {
            "query": RunnablePassthrough(),
            "profiles": lambda q: format_docs(retriever.invoke(q)),

        }
        | GENERATE_PROMPT
        | llm
        | StrOutputParser()
    )

    # Run
    response = chain.invoke(question)
    return {"messages": [response]}


def build_agent(memory):
    return create_react_agent(model=llm, tools=[retrieve], checkpointer=memory)

def generate_query_or_respond(state: State,agent_executor=None):
    """Call the agent executor and return only the last message."""
    result = agent_executor.invoke(
        {"messages": state["messages"]},
        config={
            "configurable": {
                "thread_id": state["thread_id"]
            }
        }
    )
    last_message = result["messages"][-1]
    return {"messages": [last_message]}


