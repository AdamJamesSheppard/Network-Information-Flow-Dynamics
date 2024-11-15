import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.animation import FuncAnimation

# Create a social network graph
def create_social_network(num_agents, avg_connections):
    G = nx.erdos_renyi_graph(n=num_agents, p=avg_connections/num_agents)
    return G

# Initialize the states of agents
def initialize_agents(G, initial_informed):
    nx.set_node_attributes(G, 'unaware', 'state')
    initial_informed_agents = random.sample(list(G.nodes()), initial_informed)
    for agent in initial_informed_agents:
        G.nodes[agent]['state'] = 'informed'
    return G, initial_informed_agents

# Simulate the information diffusion process and collect states over time
def simulate_diffusion(G, infection_prob, steps):
    informed_counts = []
    states_over_time = []

    for _ in range(steps):
        new_informed = []
        for node in G.nodes():
            if G.nodes[node]['state'] == 'informed':
                neighbors = list(G.neighbors(node))
                for neighbor in neighbors:
                    if G.nodes[neighbor]['state'] == 'unaware' and random.random() < infection_prob:
                        new_informed.append(neighbor)
        for agent in new_informed:
            G.nodes[agent]['state'] = 'informed'
        informed_counts.append(sum(1 for node in G.nodes() if G.nodes[node]['state'] == 'informed'))
        states_over_time.append([G.nodes[node]['state'] for node in G.nodes()])
    return informed_counts, states_over_time

# Logistic function for curve fitting
def logistic_function(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

# Fit logistic curve to the diffusion data
def fit_logistic_curve(informed_counts, steps):
    t = np.arange(steps)
    L = max(informed_counts)
    popt, _ = curve_fit(logistic_function, t, informed_counts, p0=[L, 0.1, steps/2])
    return popt

# Generate states over time using the approximated function
def generate_approximated_states(G, L, k, t0, steps):
    states_over_time = []
    for t in range(steps):
        informed_count = logistic_function(t, L, k, t0)
        informed_agents = random.sample(list(G.nodes()), int(informed_count))
        for node in G.nodes():
            G.nodes[node]['state'] = 'informed' if node in informed_agents else 'unaware'
        states_over_time.append([G.nodes[node]['state'] for node in G.nodes()])
    return states_over_time

# Kalman filter update function
def kalman_filter(predicted, observed, Q, R):
    # Prediction update
    predicted_state = predicted[0]
    predicted_cov = predicted[1] + Q
    
    # Measurement update
    K = predicted_cov / (predicted_cov + R)
    estimated_state = predicted_state + K * (observed - predicted_state)
    estimated_cov = (1 - K) * predicted_cov
    
    return estimated_state, estimated_cov

# Function to predict the next state
def predict_next_state(current_state, current_cov, Q):
    predicted_state = current_state
    predicted_cov = current_cov + Q
    return predicted_state, predicted_cov

# Predict the next node to be infected
def predict_next_nodes(G, current_states, infection_prob):
    next_informed = []
    for node in G.nodes():
        if current_states[node] == 'informed':
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                if current_states[neighbor] == 'unaware' and random.random() < infection_prob:
                    next_informed.append(neighbor)
    return next_informed

# Parameters
num_agents = 100
avg_connections = 5
initial_informed = 5
infection_prob = 0.1
steps = 20
n_future_steps = 5

# Create and initialize the network
G = create_social_network(num_agents, avg_connections)
G, initial_informed_agents = initialize_agents(G, initial_informed)

# Simulate the initial diffusion process
informed_counts, initial_states_over_time = simulate_diffusion(G, infection_prob, steps)

# Fit a logistic curve to the diffusion data
L, k, t0 = fit_logistic_curve(informed_counts, steps)

# Initialize the Kalman filter
Q = 1  # Process noise covariance
R = 2  # Measurement noise covariance
predicted_state = informed_counts[0]
predicted_cov = 1

# Create a new network for Kalman filtered diffusion with the same structure
G2 = G.copy()

# Generate states over time using the Kalman filter
kalman_filtered_counts = []
for t in range(steps):
    # Get the logistic model prediction
    logistic_predicted = logistic_function(t, L, k, t0)
    
    # Apply the Kalman filter
    observed = informed_counts[t]
    estimated_state, estimated_cov = kalman_filter((predicted_state, predicted_cov), observed, Q, R)
    
    # Update the predicted state and covariance
    predicted_state, predicted_cov = estimated_state, estimated_cov
    
    # Store the filtered counts
    kalman_filtered_counts.append(estimated_state)

# Advance the Kalman filter by n_future_steps without new observations
future_kf_predictions = []
for _ in range(n_future_steps):
    predicted_state, predicted_cov = predict_next_state(predicted_state, predicted_cov, Q)
    future_kf_predictions.append(predicted_state)

# Advance the ABM by n_future_steps to get the actual future states
future_abm_counts, future_abm_states = simulate_diffusion(G, infection_prob, n_future_steps)

# Predict the next nodes to be infected
future_kf_nodes = []
current_states = initial_states_over_time[-1]
for _ in range(n_future_steps):
    next_informed = predict_next_nodes(G, current_states, infection_prob)
    future_kf_nodes.append(next_informed)
    for node in next_informed:
        current_states[node] = 'informed'

# Animation function for diffusion
def animate_diffusion(i, G, pos, states_over_time, future_states, future_kf_predictions, future_kf_nodes, ax, title):
    ax.clear()
    color_map = {'unaware': 'red', 'informed': 'green', 'predicted': 'blue'}
    if i < len(states_over_time):
        colors = [color_map[state] for state in states_over_time[i]]
    else:
        colors = [color_map['predicted'] if node in future_kf_nodes[i - len(states_over_time)] else color_map['unaware'] if state == 'unaware' else color_map['informed'] for node, state in enumerate(future_states[i - len(states_over_time)])]
    nx.draw(G, pos, node_color=colors, with_labels=True, ax=ax)
    if i >= len(states_over_time):
        ax.set_title(f"{title} - Step {i+1} (Future, KF: {future_kf_predictions[i - len(states_over_time)]:.0f})")
    else:
        ax.set_title(f"{title} - Step {i+1}")

# Create the animations side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
pos = nx.spring_layout(G)

def update(i):
    animate_diffusion(i, G, pos, initial_states_over_time, future_abm_states, future_kf_predictions, future_kf_nodes, ax1, "ABM Diffusion")
    animate_diffusion(i, G2, pos, initial_states_over_time, future_abm_states, future_kf_predictions, future_kf_nodes, ax2, "Kalman Filtered Diffusion")

ani = FuncAnimation(fig, update, frames=steps + n_future_steps, repeat=False)
plt.show()

# Update plot to include future predictions
plt.figure()
plt.plot(range(steps), informed_counts, label="Initial Diffusion")
plt.plot(range(steps), kalman_filtered_counts, label="Kalman Filtered Diffusion", linestyle='dashed')
plt.plot(range(steps, steps + n_future_steps), future_kf_predictions, label="Future KF Predictions", linestyle='dotted', color='blue')
plt.plot(range(steps, steps + n_future_steps), future_abm_counts, label="Future ABM Counts", linestyle='dashed', color='red')
plt.xlabel('Time Steps')
plt.ylabel('Number of Informed Agents')
plt.legend()
plt.title('Comparison of Information Diffusion and Future Predictions')
plt.show()

# Print the error for the observed period
mse = np.mean((np.array(informed_counts) - np.array(kalman_filtered_counts)) ** 2)
print(f"Mean Squared Error between initial and Kalman filtered diffusion: {mse:.2f}")
