import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to simulate information propagation with agreement/disagreement on a graph
def simulate_propagation(G, source_node, initial_state, num_steps, analysis_steps, alpha=0.1):
    current_state = initial_state.copy()
    states = [current_state]
    for step in range(num_steps):
        new_state = current_state.copy()
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors:
                influence = np.mean([current_state[neighbor] for neighbor in neighbors])
                influence_noise = np.random.normal(0, 0.5)  # Influence drawn from a normal distribution
                new_state[node] += alpha * influence_noise * influence
        
        # Apply the analysis effect at the specified steps
        if step in analysis_steps:
            for node in G.nodes():
                adjustment = np.random.normal(0, 0.5)  # Adjust by a normal distribution centered around 0
                new_state[node] += adjustment
        
        states.append(new_state)
        current_state = new_state
    return states

# Create a sample graph
G = nx.karate_club_graph()
source_nodes = [0, 10, 20]  # Choose nodes to start the information propagation
num_steps = 50
analysis_steps = [15, 30, 45]  # Steps at which the analysis is introduced

# Initialize the state of the nodes
initial_state = np.zeros(len(G.nodes()))

# Track the final state of each node before each policy is introduced
final_states = []

# Simulate propagation for each policy
combined_states = []
current_state = initial_state.copy()
policy_start_steps = []
for i, source_node in enumerate(source_nodes):
    # Set the initial state for agreement and disagreement
    agreement_nodes = {source_node}  # Start with the source node agreeing
    disagreement_nodes = set(np.random.choice(list(set(G.nodes()) - agreement_nodes), size=len(G.nodes())//4, replace=False))  # Randomly choose nodes to disagree

    for node in disagreement_nodes:
        current_state[node] = -1.0  # Disagreement state
    current_state[source_node] = 1.0  # Agreement state

    states = simulate_propagation(G, source_node, current_state, num_steps, analysis_steps)
    final_states.append(states[-1])
    current_state = states[-1].copy()  # Update the initial state for the next policy
    combined_states.extend(states)
    policy_start_steps.append(i * num_steps)

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 8))
pos = nx.spring_layout(G)

# Create the color map and normalization
cmap = plt.cm.coolwarm
norm = plt.Normalize(vmin=-2, vmax=2)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Add the color bar to the plot
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Agreement / Disagreement Score')

# Function to update the plot for animation
def update(num, G, pos, combined_states, ax, final_states, source_nodes, policy_start_steps):
    ax.clear()
    state_index = num
    nx.draw(G, pos, node_color=combined_states[state_index], with_labels=True, node_size=700, cmap=cmap, ax=ax, vmin=-2, vmax=2)
    # Highlight the source nodes
    for i, source_node in enumerate(source_nodes):
        if policy_start_steps[i] <= state_index < policy_start_steps[i] + num_steps:
            nx.draw_networkx_nodes(G, pos, nodelist=[source_node], node_color='yellow', node_size=900, ax=ax)
    ax.set_title(f'Step {num}')
    if state_index >= (len(combined_states) // 3) * 3 - 1:
        for node, (x, y) in pos.items():
            ax.text(x, y + 0.1, f'{final_states[-1][node]:.2f}', fontsize=8, ha='center')
    cbar.update_normal(sm)
    fig.suptitle('Agreement and Disagreement Propagation Across Policies', fontsize=16)

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(combined_states), fargs=(G, pos, combined_states, ax, final_states, source_nodes, policy_start_steps), interval=200, repeat=False)

# Display the animation
plt.show()

# Output final states for each policy
for i, final_state in enumerate(final_states):
    print(f"Final state after policy {i+1}:")
    print(final_state)
