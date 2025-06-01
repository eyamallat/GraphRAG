from neo4j import GraphDatabase
from networkx import DiGraph
import groq
import os
import networkx as nx
from dotenv import load_dotenv


def initialiseNeo4jSchema(kg: nx.DiGraph):
    """Initialize Neo4j schema and store knowledge graph."""
    load_dotenv()

    driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"), 
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
    cypher_schema = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Section) REQUIRE (c.key) IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.key) IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Document) REQUIRE (c.url_hash) IS UNIQUE;",
        "CREATE VECTOR INDEX `chunkVectorIndex` IF NOT EXISTS FOR (e:Embedding) ON (e.value) "
        "OPTIONS { indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};"
    ]
    with driver.session() as session:
        for cypher in cypher_schema:
            session.run(cypher)
        # Create nodes
        for node, attrs in kg.nodes(data=True):
            node_id=node
            node_type = attrs.get("group", "UnknownType")  
            node_props = {"id": node_id, **attrs}
              
            query=f"""
            MERGE (n:{node_type} {{id: $id}}) 
            SET n += $props
            """
            session.run(
                query,
                id=node_id,
                props=node_props,
            )

        # Create edges
        for source, target, attrs in kg.edges(data=True):
            relationship = attrs.get("relationship", "unknown").replace(" ", "_")

            session.run(
                f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                MERGE (source)-[:{relationship}]->(target)
                """,
                source_id=source,
                target_id=target,
            )

    driver.close()
    print("Neo4j schema initialized and knowledge graph stored successfully.")
      
class Neo4jGraph():
    
    def __init__(self):
        self.driver = GraphDatabase.driver(os.getenv("NEO4J_URI"),auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
        with self.driver.session() as session:
            session.run("RETURN 1")
        print("Successfully connected to Neo4j database")
        self.groq_client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))
        
    def query_graph(self, question):
        """Convert a natural language question into a Cypher query and fetch results."""
        
        cypher_query = self.generate_cypher_query(question)

        if not cypher_query:
            return "I couldn't generate a query for that. Try rephrasing your question."

        print(f"Generated Cypher Query: {cypher_query}")

        with self.driver.session() as session:   
            result = session.run(cypher_query)
            data = [record.data() for record in result]

        return self.format_response(question, data)
    
    def generate_cypher_query(self, question):
        # Define the schema context for the Neo4j graph
        schema_context = """
            Neo4j Knowledge Graph Schema for Candidate Data

            Node Labels & Properties:
            - person:       {id, label (name), name,email, years_of_experience}
            - university:   {id,name, label (name)}
            - degree:       {id, label (degree name),name, graduation_date}
            - company:      {id,name, label (company name)}
            - role:         {id, label (job title), name,start_date, end_date, achievements}
            - skill:        {id, label (skill name),name, level}
            - language:     {id, label (language name), name,proficiency}
            - project:      {id, label (project name), name,project_description}
            - mission:      {id, label (mission title), name,description, company}

            Relationships:
            - (person)-[:STUDIED_AT]->(university)
            - (university)-[:OFFERS]->(degree)
            - (person)-[:OBTAINED]->(degree)
            - (person)-[:WORKED_AT]->(company)
            - (company)-[:OFFERED]->(role)
            - (person)-[:PERFORMED]->(role)
            - (person)-[:HAS_SKILL]->(skill)
            - (person)-[:SPEAKS]->(language)
            - (person)-[:WORKED_ON]->(project)
            - (person)-[:ASSIGNED_TO]->(mission)
        """

        # Create the prompt to generate the Cypher query
        prompt = f"""
            You are a Cypher expert. Given the following schema and a natural language question, generate a **valid Cypher query**.

            {schema_context}

            Question: "{question}"
            
            Guidelines:
            
            1. Use .toLower() when matching names or labels (e.g., toLower(skill.name) or toLower(person.name)).
            2. Use case-insensitive fuzzy search with toLower() and CONTAINS for partial matches.
            3. Always use correct node labels and relationship types from the schema (person, skill, company, project, etc.).
            4. Use OPTIONAL MATCH for optional relationships (e.g., certifications).
            5. Return only relevant properties with meaningful aliases using AS (e.g., person.name AS personName).
            6. Do NOT wrap the Cypher query in triple backticks or Markdown formatting.
            7. Return only the raw Cypher query – no explanation, comments, or formatting around it.

            Generate only the Cypher query below:
        """

        # Request the Cypher query from the groq_client API
        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            temperature=0.2
        )

        # Extract the generated Cypher query from the response
        cypher_query = response.choices[0].message.content.strip()

        return cypher_query

    
    
    def format_response(self, question, data):
        """Generate a natural language response from the query results."""
        if not data:
            return "I couldn't find any information matching your question in the database."
        #i  can limit data to avoid token limits
        
        prompt = f"""
        You are an AI assistant. Based on the following extracted data, answer the question naturally.
        Question: "{question}"
        Data: {data}
        
        Guidelines:
        1. Provide a direct, clear answer to the question
        2. Highlight key findings or patterns in the data
        3. If there are many results, summarize them rather than listing all
        4. Use natural language rather than technical database terminology
        5. Format dates, names, and numbers appropriately
        6. For lists of people or entities, use a clear structure
        7. If the results show connections or relationships, explain them clearly.
        """

        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )

        return response.choices[0].message.content.strip()

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()