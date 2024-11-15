import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation

# Create a social network
def create_social_network(num_nodes, probability):
    G = nx.erdos_renyi_graph(num_nodes, probability)
    return G

# Initialize agents with SIR model states
def initialize_agents(G, initial_infected=1):
    states = {}
    for node in G.nodes:
        if node < initial_infected:
            states[node] = 'I'  # Initial infected nodes
        else:
            states[node] = 'S'  # Susceptible nodes
    return states

# Define the SIR model dynamics
def sir_model(states, G, beta, gamma):
    new_states = states.copy()
    for node in G.nodes:
        if states[node] == 'S':
            for neighbor in G.neighbors(node):
                if states[neighbor] == 'I' and random.random() < beta:
                    new_states[node] = 'I'
                    break
        elif states[node] == 'I':
            if random.random() < gamma:
                new_states[node] = 'R'
    return new_states

# Simulation parameters
num_nodes = 30
probability = 0.1
initial_infected = 3
beta = 0.3  # Infection rate
gamma = 0.1  # Recovery rate
steps = 100
t = np.linspace(0, 10, steps)

# Create and simulate the network
G = create_social_network(num_nodes, probability)
states = initialize_agents(G, initial_infected)

# Simulation function
def simulate(states, G, beta, gamma):
    new_states = states.copy()
    for _ in range(steps):
        new_states = sir_model(new_states, G, beta, gamma)
        yield new_states

# Visualization setup
fig, ax = plt.subplots(figsize=(12, 8))
pos = nx.spring_layout(G)

def update(num):
    ax.clear()
    states = next(simulation)
    susceptible_nodes = [node for node, state in states.items() if state == 'S']
    infected_nodes = [node for node, state in states.items() if state == 'I']
    recovered_nodes = [node for node, state in states.items() if state == 'R']
    
    nx.draw_networkx_nodes(G, pos, nodelist=susceptible_nodes, node_color='blue', label='Susceptible', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=infected_nodes, node_color='red', label='Infected', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=recovered_nodes, node_color='green', label='Recovered', ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)
    ax.set_title(f'Time {t[num]:.2f}')
    ax.legend()

simulation = simulate(states, G, beta, gamma)
ani = FuncAnimation(fig, update, frames=len(t), repeat=False)
plt.show()
