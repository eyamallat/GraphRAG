from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_mistralai import ChatMistralAI
from vectorstore.rag_pipeline import create_retriever, format_docs
import os
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

retriever=create_retriever()
GENERATE_PROMPT = PromptTemplate.from_template(
        """
    Human: You are an AI assistant helping evaluate candidate profiles based on a job requirement.

    Given the following candidate profiles:
    {profiles}

    And the following job requirement or query:
    {query}

    Your tasks:
        -Rank the candidates from most to least relevant to the job and refer to them by their ids
        - Assign a relevance score to each candidate from 0 (not relevant) to 10 (highly relevant).
        - Only include candidates with a score strictly greater than 4 in your answer.
        - Use specific details (e.g., job titles, skills, experiences) to support your ranking.
        - Don't show the irrelevant candidates in the answer

    Return the answer clearly and concisely, in **French**.
    Assistant:

    """
)

llm = ChatMistralAI(model="mistral-large-latest", temperature=0, api_key=os.environ["MISTRAL_API_KEY"])  
retrieve= create_retriever_tool(
    retriever,
    "retrieve_candidates",
    "Ranks and scores candidates based on job requirements",
)

def generate_answer(state: MessagesState):
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




def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        llm
        .bind_tools([retrieve]).invoke(state["messages"])
    )
    return {"messages": [response]}
