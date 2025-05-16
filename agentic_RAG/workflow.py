from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from agentic_RAG.components import generate_answer,generate_query_or_respond,retrieve
from langgraph.graph import MessagesState

workflow = StateGraph(MessagesState)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retrieve]))
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)
workflow.add_edge("retrieve","generate_answer")
workflow.add_edge("generate_answer", END)


# Compile
graph = workflow.compile()