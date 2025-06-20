import uuid
from agentic_RAG.workflow import build_graph
from pymilvus import connections
from langchain_core.messages import HumanMessage,AIMessage
from pprint import pprint
import asyncio
import sys
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
import os


async def main():
    dsn = (
            f"postgres://{os.getenv('PSQL_USERNAME')}:{os.getenv('PSQL_PASSWORD')}"
            f"@{os.getenv('PSQL_HOST')}:{os.getenv('PSQL_PORT')}/{os.getenv('PSQL_DATABASE')}"
    )
    sslmode = os.getenv('PSQL_SSLMODE')
    if sslmode:
        dsn += f"?sslmode={sslmode}"

    async with AsyncConnectionPool(
        conninfo=dsn,
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
        }
    ) as pool:
        async with pool.connection() as conn:
            memory = AsyncPostgresSaver(conn)
            await memory.setup()
            graph = build_graph(memory=memory)
    
            print(" LangGraph Agentic RAG is live.")
            thread_id = str(uuid.uuid4()) 
            print(thread_id)
            config = {}
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
                        ],
                        "thread_id":thread_id
                    }
                    result=await graph.ainvoke(
                        initial_state,
                        config={
                        "configurable": {
                            "thread_id": initial_state["thread_id"]  
                            }
                        }
    
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
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
