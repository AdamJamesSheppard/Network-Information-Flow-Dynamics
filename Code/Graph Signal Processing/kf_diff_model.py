import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to initialize the ensemble
def initialize_ensemble(G, initial_state, ensemble_size=10):
    ensemble = [initial_state + np.random.normal(0, 0.1, size=len(G.nodes())) for _ in range(ensemble_size)]
    return np.array(ensemble)

# Nonlinear propagation function
def nonlinear_propagation(G, state, alpha=0.1):
    new_state = state.copy()
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:
            influence = np.tanh(np.mean([state[neighbor] for neighbor in neighbors]))  # Nonlinear influence using tanh
            influence_noise = np.random.normal(0, 0.5)  # Influence drawn from a normal distribution
            new_state[node] += alpha * influence_noise * influence
    return new_state

# Function to perform data assimilation using Ensemble Kalman Filter (EnKF)
def ensemble_kalman_filter(ensemble, observations, observation_noise):
    ensemble_mean = np.mean(ensemble, axis=0)
    ensemble_perturbations = ensemble - ensemble_mean

    observation_ensemble = ensemble + np.random.normal(0, observation_noise, ensemble.shape)
    observation_mean = np.mean(observation_ensemble, axis=0)
    observation_perturbations = observation_ensemble - observation_mean

    cross_covariance = np.dot(ensemble_perturbations.T, observation_perturbations) / (ensemble.shape[0] - 1)
    observation_covariance = np.dot(observation_perturbations.T, observation_perturbations) / (ensemble.shape[0] - 1) + observation_noise * np.eye(observations.shape[0])

    kalman_gain = np.dot(cross_covariance, np.linalg.inv(observation_covariance))

    analysis_ensemble = ensemble + np.dot(kalman_gain, (observations - observation_mean).T).T

    return analysis_ensemble

# Function to simulate information propagation with agreement/disagreement using EnKF
def simulate_propagation_with_enkf(G, source_node, initial_state, num_steps, analysis_steps, alpha=0.1, ensemble_size=10):
    ensemble = initialize_ensemble(G, initial_state, ensemble_size)
    states = [np.mean(ensemble, axis=0)]
    
    for step in range(num_steps):
        ensemble = np.array([nonlinear_propagation(G, state, alpha) for state in ensemble])
        
        if step in analysis_steps:
            observations = np.random.normal(0, 0.5, size=len(G.nodes()))
            ensemble = ensemble_kalman_filter(ensemble, observations, observation_noise=0.1)
        
        states.append(np.mean(ensemble, axis=0))
        
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

    states = simulate_propagation_with_enkf(G, source_node, current_state, num_steps, analysis_steps)
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
