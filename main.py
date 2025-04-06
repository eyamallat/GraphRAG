import os
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
from networkx import DiGraph
import groq
import os
from knowledge_graph.data_loader import load_data
from knowledge_graph.graph_builder import create_knowledge_graph
from knowledge_graph.graph_visualizer import visualize_graph
from knowledge_graph.neo4j_handler import initialiseNeo4jSchema,Neo4jGraph
from dotenv import load_dotenv

def main():
    
    load_dotenv()
    
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
    groq_client = groq.Client(api_key=GROQ_API_KEY)
    
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"), 
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )

    graph = Neo4jGraph()
    file_path = '/app/candidates.json'
    
    try:
        data = load_data(file_path)
        G = create_knowledge_graph(data)
        visualize_graph(G)
        initialiseNeo4jSchema(G)
        
        print(f"Knowledge graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Count nodes by type
        node_types = {}
        for node, attrs in G.nodes(data=True):
            group = attrs.get('group', 'unknown')
            node_types[group] = node_types.get(group, 0) + 1
        
        print("\nNode types:")
        for node_type, count in node_types.items():
            print(f" - {node_type}: {count}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        
    try:
        while True:
            question = input("\nAsk a question (or type 'exit' to quit): ")
            if question.lower() == "exit":
                break

            answer = graph.query_graph(question)
            print("\nAnswer:", answer)
    
    finally:
        graph.close()

if __name__ == "__main__":
    main()