import networkx as nx
import matplotlib.pyplot as plt
import random
import time

# Step 1: Create a social network graph
def create_social_network(num_nodes, prob_edge):
    G = nx.erdos_renyi_graph(num_nodes, prob_edge)
    return G

# Step 2: Simulate information diffusion
def simulate_diffusion(G, initial_infected, steps):
    status = {node: 'S' for node in G.nodes()}  # S for susceptible, I for infected
    for node in initial_infected:
        status[node] = 'I'
    
    infected_nodes = [initial_infected]
    
    for step in range(steps):
        new_infected = []
        for node in infected_nodes[-1]:
            for neighbor in G.neighbors(node):
                if status[neighbor] == 'S' and random.random() < 0.5:  # 0.5 is the infection probability
                    status[neighbor] = 'I'
                    new_infected.append(neighbor)
        infected_nodes.append(new_infected)
    
    return infected_nodes

# Step 3: Visualize the diffusion process
def visualize_diffusion(G, infected_nodes):
    pos = nx.spring_layout(G)  # Positioning the nodes with spring layout
    plt.ion()
    
    for step, nodes in enumerate(infected_nodes):
        plt.clf()
        node_colors = ['red' if node in sum(infected_nodes[:step+1], []) else 'blue' for node in G.nodes()]
        nx.draw(G, pos, node_color=node_colors, with_labels=True)
        plt.title(f'Step {step}')
        plt.show()
        plt.pause(1)
    
    plt.ioff()
    plt.show()

# Parameters
num_nodes = 20
prob_edge = 0.2
initial_infected = [0]
steps = 5

# Run the simulation and visualization
G = create_social_network(num_nodes, prob_edge)
infected_nodes = simulate_diffusion(G, initial_infected, steps)
visualize_diffusion(G, infected_nodes)
