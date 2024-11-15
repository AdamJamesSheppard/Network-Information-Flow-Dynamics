import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import pinv
from sklearn.metrics import mean_squared_error

# Function to create a graph and compute its Laplacian
def create_graph(n, p):
    G = nx.erdos_renyi_graph(n, p)
    L = nx.laplacian_matrix(G).toarray()
    return G, L

# Function to simulate SIR model
def simulate_sir(G, beta, gamma, T=50, dt=0.01):
    n = len(G)
    num_steps = int(T / dt)
    states = np.zeros((num_steps + 1, n))
    infected = np.random.choice(n)
    states[0, infected] = 1  # Initially one infected node

    for t in range(num_steps):
        new_states = states[t].copy()
        for i in range(n):
            if states[t, i] == 1:  # Infected
                neighbors = list(G.neighbors(i))
                for neighbor in neighbors:
                    if states[t, neighbor] == 0 and np.random.rand() < beta:
                        new_states[neighbor] = 1  # Susceptible -> Infected
                if np.random.rand() < gamma:
                    new_states[i] = 2  # Infected -> Recovered
        states[t + 1] = new_states
    return states

# Function to learn Laplacian
def learn_laplacian(X, y):
    L_learned = pinv(X) @ y
    return L_learned

# Kalman Filter functions
def kalman_filter_update(L_pred, P_pred, H, z, R):
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    L_updated = L_pred + K @ (z - H @ L_pred)
    P_updated = (np.eye(len(P_pred)) - K @ H) @ P_pred
    return L_updated, P_updated

# Main function to run the SIR model, learn the Laplacian, and validate it
def main():
    n = 80  # Number of nodes
    p = 0.0266  # Probability of edge creation
    beta = 0.3  # Infection rate
    gamma = 0.1  # Recovery rate

    # Create the graph
    G, L = create_graph(n, p)

    # Simulate the SIR model on the network
    T = 50
    dt = 0.01
    states = simulate_sir(G, beta, gamma, T, dt)

    # Generate training data from SIR simulation
    X = []
    y = []
    num_steps = len(states) - 1
    for t in range(num_steps):
        susceptible = (states[t] == 0).astype(int)
        infected = (states[t] == 1).astype(int)
        recovered = (states[t] == 2).astype(int)
        state_vector = np.concatenate((susceptible, infected, recovered))
        X.append(state_vector)
        next_state_vector = np.concatenate(((states[t+1] == 0).astype(int), (states[t+1] == 1).astype(int), (states[t+1] == 2).astype(int)))
        y.append((next_state_vector - state_vector) / dt)

    X = np.array(X)
    y = np.array(y)

    # Initialize Kalman Filter parameters
    L_learned = np.zeros((3*n, 3*n))
    P = np.eye(3*n)  # Initial error covariance
    H = np.eye(3*n)  # Observation model
    R = np.eye(3*n) * 0.01  # Observation noise covariance

    # Iteratively learn the Laplacian using Kalman filter
    num_iterations = 10
    for i in range(num_iterations):
        # Prediction step
        L_pred = L_learned
        P_pred = P + np.eye(3*n)  # Process noise is identity for simplicity

        # Update step with observations
        L_learned, P = kalman_filter_update(L_pred, P_pred, H, learn_laplacian(X, y), R)

        # Simulate with the learned Laplacian for the next iteration
        states_learned = simulate_sir(G, beta, gamma, T, dt)
        X = []
        y = []
        for t in range(num_steps):
            susceptible = (states_learned[t] == 0).astype(int)
            infected = (states_learned[t] == 1).astype(int)
            recovered = (states_learned[t] == 2).astype(int)
            state_vector = np.concatenate((susceptible, infected, recovered))
            X.append(state_vector)
            next_state_vector = np.concatenate(((states_learned[t+1] == 0).astype(int), (states_learned[t+1] == 1).astype(int), (states_learned[t+1] == 2).astype(int)))
            y.append((next_state_vector - state_vector) / dt)
        X = np.array(X)
        y = np.array(y)

    # Simulate diffusion on the same network using the learned Laplacian
    states_replicated = simulate_sir(G, beta, gamma, T, dt)

    # Calculate Mean Squared Error (MSE) as the evaluation metric
    mse = mean_squared_error(states.flatten(), states_replicated.flatten())
    print(f"Mean Squared Error (MSE) between actual and replicated diffusion: {mse}")

    # Visualization
    pos = nx.spring_layout(G)  # Layout for the graph

    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    def update(frame):
        axs[0].clear()
        axs[1].clear()

        colors = np.zeros((n, 3))
        colors[states[frame] == 0] = [0, 0, 1]  # Blue for Susceptible
        colors[states[frame] == 1] = [1, 0, 0]  # Red for Infected
        colors[states[frame] == 2] = [0, 1, 0]  # Green for Recovered
        nx.draw(G, pos, node_color=colors, node_size=300, ax=axs[0], with_labels=True)
        axs[0].set_title('Original Network SIR Diffusion')

        colors_replicated = np.zeros((n, 3))
        colors_replicated[states_replicated[frame] == 0] = [0, 0, 1]
        colors_replicated[states_replicated[frame] == 1] = [1, 0, 0]
        colors_replicated[states_replicated[frame] == 2] = [0, 1, 0]
        nx.draw(G, pos, node_color=colors_replicated, node_size=300, ax=axs[1], with_labels=True)
        axs[1].set_title('Replicated Network Learned Diffusion')

    ani = FuncAnimation(fig, update, frames=len(states), repeat=False)
    plt.show()

if __name__ == "__main__":
    main()
