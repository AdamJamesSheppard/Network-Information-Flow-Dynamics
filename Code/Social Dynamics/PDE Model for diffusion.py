import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Parameters
num_nodes = 70
prob_edge = 0.2
initial_infected = [0]
diffusion_coefficient = 0.1
reaction_rate = 0.1
steps = 50

# Create a social network
G = nx.erdos_renyi_graph(num_nodes, prob_edge)
adj_matrix = nx.to_numpy_array(G)

# Initialize state of nodes (0 = susceptible, 1 = infected)
state = np.zeros(num_nodes)
state[initial_infected] = 1

# Reaction term
def reaction(u):
    return reaction_rate * u * (1 - u)  # Example logistic growth

# Simulate diffusion
history = [state.copy()]
for step in range(steps):
    laplacian = adj_matrix @ state - state * adj_matrix.sum(axis=1)
    state += diffusion_coefficient * laplacian + reaction(state)
    state = np.clip(state, 0, 1)  # Ensure state values remain between 0 and 1
    history.append(state.copy())

# Visualize diffusion process
def visualize_diffusion(history, G):
    pos = nx.spring_layout(G)
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    
    for step, state in enumerate(history):
        ax.clear()  # Clear the plot for updating
        node_colors = ['red' if state[i] > 0.5 else 'blue' for i in range(len(state))]
        nx.draw(G, pos, node_color=node_colors, with_labels=True, ax=ax)
        plt.title(f'Step {step}')
        plt.draw()
        plt.pause(0.1)
    
    plt.ioff()  # Turn off interactive mode
    plt.show()

# Run visualization
visualize_diffusion(history, G)
