from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from agentic_RAG.components import generate_answer,generate_query_or_respond,retrieve,build_agent,State
from langgraph.graph import MessagesState



def build_graph(memory):
    agent_executor = build_agent(memory)
    
    def _query_or_respond(state: State):
        return generate_query_or_respond(state, agent_executor)
    
    workflow = StateGraph(State)

    workflow.add_node("generate_query_or_respond", _query_or_respond)
    
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
    return workflow.compile(checkpointer=memory)