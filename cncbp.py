import matplotlib.pyplot as plt
import networkx as nx
from queue import PriorityQueue
import tkinter as tk
from tkinter import simpledialog


def dijkstra(graph, source, destination):
    distances = {node: float('inf') for node in graph.nodes}
    distances[source] = 0
    predecessors = {node: None for node in graph.nodes}

    pq = PriorityQueue()
    pq.put((0, source))

    routing_table_updates = []

    while not pq.empty():
        (current_distance, current_node) = pq.get()

        if current_distance > distances[current_node]:
            continue

        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]['weight']
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                pq.put((distance, neighbor))

                routing_table_updates.append((current_node, neighbor, distances.copy()))

        if current_node == destination:
            break

    path = []
    current = destination
    while current is not None:
        path.insert(0, current)
        current = predecessors[current]

    return distances, predecessors, path, routing_table_updates

def visualize_routing_animated(graph, source, destination, routing_steps):
    pos = nx.spring_layout(graph)  
    fig, ax = plt.subplots(figsize=(10, 7))

    def draw_graph(step=None, current_edge=None, distances=None):
        ax.clear()
        nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", font_weight="bold", ax=ax)

        edge_labels = {(u, v): f"{graph[u][v]['weight']}" for u, v in graph.edges}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color="red", ax=ax)

        if distances:
            node_labels = {node: f"{distances[node]:.1f}" if distances[node] < float('inf') else "∞" for node in graph.nodes}
            
            label_pos = {node: (x, y + 0.05) for node, (x, y) in pos.items()}  
            nx.draw_networkx_labels(graph, label_pos, labels=node_labels, font_color="black", ax=ax, verticalalignment="bottom", font_size=10)

        if current_edge:
            nx.draw_networkx_edges(graph, pos, edgelist=[current_edge], edge_color="red", width=2, ax=ax)

        if step is not None:
            ax.set_title(f"Step {step + 1}: Edge {current_edge[0]} -> {current_edge[1]}")
        else:
            ax.set_title("Initial Graph")

    draw_graph()
    plt.pause(2) 

    for step, (current_node, neighbor, distances) in enumerate(routing_steps):
        draw_graph(step, (current_node, neighbor), distances)
        plt.pause(2)  

    ax.clear()
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray", font_weight="bold", ax=ax)

    for node in graph.nodes:
        pred = predecessors[node]
        if pred is not None:
            nx.draw_networkx_edges(graph, pos, edgelist=[(pred, node)], edge_color="blue", width=2, ax=ax)

    node_labels = {node: f"{distances[node]:.1f}" for node in graph.nodes}
    label_pos = {node: (x, y + 0.05) for node, (x, y) in pos.items()}  # Adjust vertical spacing
    nx.draw_networkx_labels(graph, label_pos, labels=node_labels, font_color="black", ax=ax, verticalalignment="bottom", font_size=10)

    edge_labels = {(u, v): f"{graph[u][v]['weight']}" for u, v in graph.edges}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color="red", ax=ax)

    ax.set_title(f"Shortest Paths from source node to all nodes")

    plt.show()

def get_user_input():
   
    root = tk.Tk()
    root.withdraw() 

    num_nodes = simpledialog.askinteger("Input", "Enter the number of nodes:")

    nodes = []
    for i in range(num_nodes):
        node = simpledialog.askstring("Input", f"Enter name of node {i+1}:")
        nodes.append(node)

    G = nx.Graph()

    num_edges = simpledialog.askinteger("Input", "Enter the number of edges:")

    for _ in range(num_edges):
        edge_input = simpledialog.askstring("Input", "Enter edge (format: node1 node2 weight):")
        u, v, weight = edge_input.split()
        weight = float(weight)
        G.add_edge(u, v, weight=weight)

    source_node = simpledialog.askstring("Input", "Enter the source node:")
    destination_node = simpledialog.askstring("Input", "Enter the destination node:")

    return G, source_node, destination_node

if __name__ == "__main__":
    G, source_node, destination_node = get_user_input()

    distances, predecessors, path, steps = dijkstra(G, source_node, destination_node)

    print(f"Shortest distance from {source_node} to {destination_node}:", distances[destination_node])
    print(f"Shortest path from {source_node} to {destination_node}:", path)

    visualize_routing_animated(G, source_node, destination_node, steps)
