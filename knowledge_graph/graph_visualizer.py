from pyvis.network import Network

def visualize_graph(G, output_file="knowledge_graph.html"):
    """Visualize the knowledge graph using Pyvis."""
    net = Network(height="750px", width="100%", bgcolor="#FFFFFF", font_color="black")

    # Define colors for each node group
    group_colors = {
        "person": "#3DA5D9",       # Blue
        "education": "#7B287D",    # Purple
        "experience": "#EA6A47",   # Orange
        "skill": "#4CB944",        # Green
        "project": "#FF6F91",      # Pink
        "mission": "#FFD166",      # Yellow
        "certification": "#D9534F",
        "language": "#B33F62",     # Red
        "unknown": "CCCCCC",
        
    }

    # Add nodes with their group-specific color
    for node_id in G.nodes():
        attrs = G.nodes[node_id]
        group = attrs.get('group', 'unknown')
        color = group_colors.get(group, group_colors["unknown"])

        net.add_node(
            node_id,
            label=attrs.get('label', node_id),
            title=attrs.get('title', ''),
            color=color
        )

    # Add edges with optional relationship title
    for source, target, data in G.edges(data=True):
        net.add_edge(
            source,
            target,
            title=data.get('relationship', '')  
        )

    # Customize the physics and appearance
    net.set_options("""
    var options = {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
                "enabled": true,
                "iterations": 1000
            }
        },
        "edges": {
            "smooth": {
                "type": "continuous",
                "forceDirection": "none"
            }
        },
        "nodes": {
            "font": {
                "size": 15
            },
            "size": 25
        }
    }
    """)

    # Save to HTML
    net.save_graph(output_file)
    print(f"Knowledge graph visualization has been saved to {output_file}")
    
    return net