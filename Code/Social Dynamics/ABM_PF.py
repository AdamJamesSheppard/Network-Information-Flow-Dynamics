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

# Particle Filter update function
def particle_filter_update(particles, weights, observed, N, epsilon=1e-8):
    # Resampling step
    weights_sum = np.sum(weights)
    if weights_sum == 0:
        weights_sum = epsilon
    weights /= weights_sum
    indices = np.random.choice(range(N), size=N, p=weights)
    particles = particles[indices]
    
    # Measurement update
    weights *= np.exp(-0.5 * ((observed - particles)**2) / epsilon)
    weights_sum = np.sum(weights)
    if weights_sum == 0:
        weights_sum = epsilon
    weights /= weights_sum
    
    return particles, weights

# Predict the next state of particles
def predict_next_particles(particles, Q):
    particles += np.random.normal(0, Q, size=particles.shape)
    return particles

# Parameters
num_agents = 100
avg_connections = 5
initial_informed = 5
infection_prob = 0.1
steps = 20
n_future_steps = 5
N = 1000  # Number of particles
Q = 1  # Process noise covariance
epsilon = 1e-2  # Small value to avoid division by zero

# Create and initialize the network for ABM
G_abm = create_social_network(num_agents, avg_connections)
G_abm, initial_informed_agents = initialize_agents(G_abm, initial_informed)

# Simulate the initial diffusion process for ABM
abm_informed_counts, abm_states_over_time = simulate_diffusion(G_abm, infection_prob, steps)

# Fit a logistic curve to the ABM diffusion data
L, k, t0 = fit_logistic_curve(abm_informed_counts, steps)

# Initialize the Particle Filter
particles = np.random.normal(abm_informed_counts[0], Q, size=N)
weights = np.ones(N) / N

# Generate states over time using the Particle Filter
pf_informed_counts = []
for t in range(steps):
    # Predict the next state of particles
    particles = predict_next_particles(particles, Q)
    
    # Apply the Particle Filter update
    observed = abm_informed_counts[t]
    particles, weights = particle_filter_update(particles, weights, observed, N, epsilon)
    
    # Estimate the state
    estimated_state = np.mean(particles)
    pf_informed_counts.append(estimated_state)

# Advance the Particle Filter by n_future_steps without new observations
future_pf_predictions = []
for _ in range(n_future_steps):
    particles = predict_next_particles(particles, Q)
    estimated_state = np.mean(particles)
    future_pf_predictions.append(estimated_state)

# Predict future ABM states for n_future_steps
future_abm_counts, future_abm_states = simulate_diffusion(G_abm, infection_prob, n_future_steps)

# Animation function for ABM diffusion
def animate_abm_diffusion(i, G_abm, pos_abm, abm_states_over_time, future_abm_states, ax):
    ax.clear()
    color_map = {'unaware': 'red', 'informed': 'green'}
    
    # ABM Diffusion
    if i < len(abm_states_over_time):
        colors_abm = [color_map[state] for state in abm_states_over_time[i]]
    else:
        colors_abm = [color_map['informed'] if state == 'informed' else color_map['unaware'] for state in future_abm_states[i - len(abm_states_over_time)]]
    nx.draw(G_abm, pos_abm, node_color=colors_abm, with_labels=True, ax=ax)
    if i >= len(abm_states_over_time):
        ax.set_title(f"ABM Diffusion - Step {i+1} (Future)")
    else:
        ax.set_title(f"ABM Diffusion - Step {i+1}")

# Animation function for Particle Filter diffusion
def animate_pf_diffusion(i, G_pf, pos_pf, pf_informed_counts, future_pf_predictions, ax):
    ax.clear()
    color_map = {'unaware': 'red', 'informed': 'green', 'predicted': 'blue'}
    
    # Particle Filter Diffusion
    if i < len(pf_informed_counts):
        logistic_pred = logistic_function(i, L, k, t0)
        informed_nodes = np.argsort(-np.array(particles))[:int(logistic_pred)]
        colors_pf = [color_map['informed'] if j in informed_nodes else color_map['unaware'] for j in range(num_agents)]
    else:
        logistic_pred = logistic_function(i, L, k, t0)
        informed_nodes = np.argsort(-np.array(particles))[:int(future_pf_predictions[i - len(pf_informed_counts)])]
        colors_pf = [color_map['informed'] if j in informed_nodes else color_map['unaware'] for j in range(num_agents)]
    nx.draw(G_pf, pos_pf, node_color=colors_pf, with_labels=True, ax=ax)
    if i >= len(pf_informed_counts):
        ax.set_title(f"Particle Filter Diffusion - Step {i+1} (Future)")
    else:
        ax.set_title(f"Particle Filter Diffusion - Step {i+1}")

# Create a new network for Particle Filter to ensure independence
G_pf = create_social_network(num_agents, avg_connections)
pos_abm = nx.spring_layout(G_abm)
pos_pf = nx.spring_layout(G_pf)

# Create the figure and axes for animations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Create the ABM animation
ani_abm = FuncAnimation(fig, animate_abm_diffusion, fargs=(G_abm, pos_abm, abm_states_over_time, future_abm_states, ax1), frames=steps + n_future_steps, repeat=False)

# Create the Particle Filter animation
ani_pf = FuncAnimation(fig, animate_pf_diffusion, fargs=(G_pf, pos_pf, pf_informed_counts, future_pf_predictions, ax2), frames=steps + n_future_steps, repeat=False)

plt.show()

# Update plot to include future predictions
plt.figure()
plt.plot(range(steps), abm_informed_counts, label="ABM Diffusion")
plt.plot(range(steps), pf_informed_counts, label="Particle Filter Diffusion", linestyle='dashed')
plt.plot(range(steps, steps + n_future_steps), future_pf_predictions, label="Future PF Predictions", linestyle='dotted', color='blue')
plt.plot(range(steps, steps + n_future_steps), future_abm_counts, label="Future ABM Counts", linestyle='dashed', color='red')
plt.xlabel('Time Steps')
plt.ylabel('Number of Informed Agents')
plt.legend()
plt.title('Comparison of Information Diffusion and Future Predictions')
plt.show()

# Print the error for the observed period
mse = np.mean((np.array(abm_informed_counts) - np.array(pf_informed_counts)) ** 2)
print(f"Mean Squared Error between initial and Particle filtered diffusion: {mse:.2f}")
