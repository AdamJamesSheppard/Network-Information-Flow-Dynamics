import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.animation import FuncAnimation
from joblib import Parallel, delayed

# Seed the random number generator for reproducibility
random.seed(42)
np.random.seed(42)

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
    new_informed_all_steps = []

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
        new_informed_all_steps.append(new_informed)
    return informed_counts, states_over_time, new_informed_all_steps

# Sigmoidal function for curve fitting
def sigmoidal_function(t, L, k1, k2, t0, t1):
    return L / (1 + np.exp(-k1 * (t - t0))) * (1 / (1 + np.exp(-k2 * (t - t1))))

# Fit sigmoidal curve to the diffusion data
def fit_sigmoidal_curve(informed_counts, steps):
    t = np.arange(steps)
    L = max(informed_counts)
    try:
        popt, _ = curve_fit(sigmoidal_function, t, informed_counts, p0=[L, 0.1, 0.1, steps/4, steps/2], maxfev=10000)
    except RuntimeError as e:
        print(f"Error fitting curve: {e}")
        popt = [L, 0.1, 0.1, steps/4, steps/2]  # Default parameters if fitting fails
    return popt

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

# Improved prediction of the next nodes to be infected
def predict_next_nodes(G, current_states, infection_prob):
    next_informed = []
    for node in G.nodes():
        if current_states[node] == 'informed':
            neighbors = list(G.neighbors(node))
            for neighbor in neighbors:
                if current_states[neighbor] == 'unaware' and random.random() < infection_prob:
                    next_informed.append(neighbor)
    return next_informed

# Adaptive Kalman filter update function
def adaptive_kalman_filter(predicted, observed, Q, R, previous_discrepancy):
    # Adjust Q and R based on previous discrepancy
    Q = Q * (1 + previous_discrepancy)
    R = R * (1 - previous_discrepancy)
    
    # Perform Kalman filter update
    estimated_state, estimated_cov = kalman_filter(predicted, observed, Q, R)
    
    # Calculate new discrepancy
    new_discrepancy = np.abs(observed - estimated_state)
    
    return estimated_state, estimated_cov, new_discrepancy

# Parameters
num_agents = 100
avg_connections = 3
initial_informed = 10
infection_prob = 0.0266  
steps = 40
n_future_steps = 5

# Create and initialize the network
G = create_social_network(num_agents, avg_connections)
G, initial_informed_agents = initialize_agents(G, initial_informed)

# Simulate the initial diffusion process
informed_counts, initial_states_over_time, abm_new_informed = simulate_diffusion(G, infection_prob, steps)

# Fit a sigmoidal curve to the diffusion data
L, k1, k2, t0, t1 = fit_sigmoidal_curve(informed_counts, steps)

# Initialize the Kalman filter
Q = 1  # Process noise covariance
R = 2  # Measurement noise covariance
predicted_state = sigmoidal_function(0, L, k1, k2, t0, t1)
predicted_cov = 1

# Create a new network for Kalman filtered diffusion with the same structure
G2 = G.copy()

# Generate states over time using the Kalman filter
kalman_filtered_counts = []
kf_states_over_time = []
kf_new_informed = []
previous_discrepancy = 0
for t in range(steps):
    # Get the logistic model prediction
    logistic_predicted = sigmoidal_function(t, L, k1, k2, t0, t1)
    
    # Apply the adaptive Kalman filter
    observed = informed_counts[t]
    estimated_state, estimated_cov, previous_discrepancy = adaptive_kalman_filter((predicted_state, predicted_cov), observed, Q, R, previous_discrepancy)
    
    # Update the predicted state and covariance
    predicted_state, predicted_cov = estimated_state, estimated_cov
    
    # Constrain the estimated state within valid bounds
    estimated_state = max(0, min(num_agents, estimated_state))
    
    # Store the filtered counts
    kalman_filtered_counts.append(estimated_state)
    
    # Generate state for the current step
    informed_agents = random.sample(list(G.nodes()), int(estimated_state))
    kf_states = ['informed' if node in informed_agents else 'unaware' for node in G.nodes()]
    kf_states_over_time.append(kf_states)
    
    # Predict the next nodes to be infected for the Kalman filter
    kf_next_informed = predict_next_nodes(G, kf_states, infection_prob)
    kf_new_informed.append(kf_next_informed)
    
    # Debugging outputs
    print(f"Step {t + 1}/{steps}")
    print(f"ABM Informed Count: {informed_counts[t]}, New Informed Nodes: {abm_new_informed[t]}")
    print(f"Logistic Model Predicted: {logistic_predicted:.2f}")
    print(f"Kalman Filter Predicted: {estimated_state:.2f}, New Informed Nodes: {kf_next_informed}")

# Advance the Kalman filter by n_future_steps without new observations
future_kf_predictions = []
for _ in range(n_future_steps):
    predicted_state, predicted_cov = predict_next_state(predicted_state, predicted_cov, Q)
    # Constrain the predicted state within valid bounds
    predicted_state = max(0, min(num_agents, predicted_state))
    future_kf_predictions.append(predicted_state)

# Advance the ABM by n_future_steps to get the actual future states
future_abm_counts, future_abm_states, future_abm_new_informed = simulate_diffusion(G, infection_prob, n_future_steps)

# Predict the next nodes to be infected
future_kf_nodes = []
current_states = initial_states_over_time[-1]
for _ in range(n_future_steps):
    next_informed = predict_next_nodes(G, current_states, infection_prob)
    future_kf_nodes.append(next_informed)
    for node in next_informed:
        current_states[node] = 'informed'

# Function to compare two arrays and calculate the matching percentage
def compare_arrays(array1, array2):
    # Ensure both arrays are of the same length for fair comparison
    if len(array1) != len(array2):
        raise ValueError("Arrays must be of the same length")

    # Calculate the number of matching elements
    matches = sum(1 for a, b in zip(array1, array2) if a == b)
    
    # Calculate the percentage of matching elements
    percentage = (matches / len(array1)) * 100
    
    return percentage

# Generate comparison percentages for each iteration
perc_array = []
for i in range(steps):
    percentage = compare_arrays(initial_states_over_time[i], kf_states_over_time[i])
    perc_array.append(percentage)

# Calculate the average percentage of matching values
average_percentage = np.mean(perc_array)

print(f"The average percentage of matching values is: {average_percentage:.2f}%")

# Compare the nodes predicted by the Kalman filter with the actual nodes across the whole process
correct_predictions = 0
total_predictions = 0
for t in range(steps):
    kf_informed_nodes = {node for node in range(num_agents) if kf_states_over_time[t][node] == 'informed'}
    abm_informed_nodes = {node for node in range(num_agents) if initial_states_over_time[t][node] == 'informed'}
    correct_predictions += len(kf_informed_nodes & abm_informed_nodes)
    total_predictions += len(kf_informed_nodes | abm_informed_nodes)

if total_predictions > 0:
    correct_percentage = (correct_predictions / total_predictions) * 100
else:
    correct_percentage = 0

print(f"Percentage of correct nodes predicted by Kalman filter across the whole process: {correct_percentage:.2f}%")

# Animation function for ABM diffusion
def animate_abm_diffusion(i, G, pos, states_over_time, future_states, ax):
    ax.clear()
    color_map = {'unaware': 'red', 'informed': 'green'}
    if i < len(states_over_time):
        colors = [color_map[state] for state in states_over_time[i]]
    else:
        colors = [color_map['unaware'] if state == 'unaware' else color_map['informed'] for state in future_states[i - len(states_over_time)]]
    nx.draw(G, pos, node_color=colors, with_labels=True, ax=ax)
    ax.set_title(f"ABM Diffusion - Step {i+1}")

# Animation function for Kalman filtered diffusion
def animate_kf_diffusion(i, G, pos, states_over_time, future_states, future_kf_predictions, future_kf_nodes, ax):
    ax.clear()
    color_map = {'unaware': 'red', 'informed': 'green', 'predicted': 'blue'}
    if i < len(states_over_time):
        colors = [color_map[state] for state in states_over_time[i]]
        ax.set_title(f"Kalman Filtered Diffusion - Step {i+1}")
    else:
        future_index = i - len(states_over_time)
        colors = [color_map['predicted'] if node in future_kf_nodes[future_index] else color_map['unaware'] if state == 'unaware' else color_map['informed'] for node, state in enumerate(future_states[future_index])]
        ax.set_title(f"Kalman Filtered Diffusion - Step {i+1} (KF: {future_kf_predictions[future_index]:.0f})")
    nx.draw(G, pos, node_color=colors, with_labels=True, ax=ax)

# Create the animations separately
fig1, ax1 = plt.subplots(figsize=(7, 7))
pos = nx.spring_layout(G)
ani_abm = FuncAnimation(fig1, animate_abm_diffusion, frames=steps + n_future_steps, fargs=(G, pos, initial_states_over_time, future_abm_states, ax1), repeat=False)

fig2, ax2 = plt.subplots(figsize=(7, 7))
ani_kf = FuncAnimation(fig2, animate_kf_diffusion, frames=steps + n_future_steps, fargs=(G2, pos, initial_states_over_time, future_abm_states, future_kf_predictions, future_kf_nodes, ax2), repeat=False)
plt.show()

# Update plot to include future predictions
plt.figure()
plt.plot(range(steps), informed_counts, label="Initial Diffusion")
plt.plot(range(steps), kalman_filtered_counts, label="Kalman Filtered Diffusion", linestyle='dashed')
plt.fill_between(range(steps), np.array(kalman_filtered_counts) - np.sqrt(predicted_cov), 
                 np.array(kalman_filtered_counts) + np.sqrt(predicted_cov), color='gray', alpha=0.5)
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
