import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation

# Define the parameters
num_devices = 100
initial_infected_fraction = 0.05
prob_transmission = 0.1
prob_recovery = 0.05
prob_reboot = 0.01
num_steps = 50

# Create a P2P network (random network for simplicity)
G = nx.erdos_renyi_graph(num_devices, 0.1)

# Get fixed positions for the nodes
pos = nx.spring_layout(G)

# Initialize the states of the devices
states = np.array(['S'] * num_devices)

# Randomly infect a fraction of the devices
initial_infected = random.sample(range(num_devices), int(initial_infected_fraction * num_devices))
for i in initial_infected:
    states[i] = 'I'

# Function to update states
def update_states(G, states, prob_transmission, prob_recovery, prob_reboot):
    new_states = states.copy()
    for node in G.nodes():
        if states[node] == 'I':
            # Infected devices have a chance to recover
            if random.random() < prob_recovery:
                new_states[node] = 'R'
            else:
                # Infected devices spread the malware to susceptible neighbors
                for neighbor in G.neighbors(node):
                    if states[neighbor] == 'S' and random.random() < prob_transmission:
                        new_states[neighbor] = 'I'
        elif states[node] == 'S':
            # Susceptible devices can reboot to clear non-persistent malware
            if random.random() < prob_reboot:
                new_states[node] = 'S'
    return new_states

# Store the state of the network at each step
state_history = [states.copy()]

for step in range(num_steps):
    states = update_states(G, states, prob_transmission, prob_recovery, prob_reboot)
    state_history.append(states.copy())

# Create a color map for the states
color_map = {'S': 'blue', 'I': 'red', 'R': 'green'}

def draw_graph(step):
    plt.clf()
    colors = [color_map[state] for state in state_history[step]]
    nx.draw(G, pos=pos, node_color=colors, with_labels=True, node_size=100)
    plt.title(f"Step {step}")

# Create the animation
fig = plt.figure(figsize=(10, 10))
ani = FuncAnimation(fig, draw_graph, frames=num_steps, interval=200, repeat=False)

# Save the animation
ani.save('botnet_spread.mp4', writer='ffmpeg', fps=5)

plt.show()
