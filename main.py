from agentic_RAG.workflow import graph
from pymilvus import connections
from langchain_core.messages import HumanMessage,AIMessage
from pprint import pprint

def main():
    #run_rag_pipeline()
    
    print(" LangGraph Agentic RAG is live.")
    thread_id = "26052025"
    config = {"configurable": {"thread_id": thread_id}}
    try:
    
        while True:
            query = input("\nAsk a question (or type 'exit' to quit): ")
            if query.lower() == "exit":
                break

            # Run the graph as a generator
            initial_state = {
                "messages": [
                    HumanMessage(content="Répondez uniquement en français."),
                    HumanMessage(content=query)
                ]
            }
            result= graph.invoke(
                initial_state,
                config,
            )
            final_message = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, (AIMessage, dict)) or hasattr(msg, 'content'):
                    final_message = msg
                    break
            print("\n🧾 Answer:\n")
            print(final_message.content) 
          
    except KeyboardInterrupt:
        print("\n[Shutdown]")
    finally:
        connections.disconnect("default")


if __name__ == "__main__":
    main()