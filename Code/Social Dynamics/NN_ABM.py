import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from matplotlib.animation import FuncAnimation

# Seed the random number generator for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("Running...")

# Define the Graph Neural Network (GNN) model
class GNN(nn.Module):
    def __init__(self, num_node_features, hidden_channels, output_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

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

# Prepare data for GNN
def prepare_gnn_data(G, states_over_time):
    node_features = []
    for step_states in states_over_time:
        features = [[1, 0] if state == 'informed' else [0, 1] for state in step_states]
        node_features.append(features)
    
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    return node_features, edge_index

# Train GNN model
def train_gnn(model, optimizer, criterion, train_data, train_targets, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(train_data)
        loss = criterion(out, train_targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Predict diffusion using trained GNN model
def predict_gnn(model, test_data):
    model.eval()
    with torch.no_grad():
        out = model(test_data)
    return out

# Parameters
num_agents = 100
avg_connections = 3
initial_informed = 10
infection_prob = 0.0266  
steps = 40
n_future_steps = 5
hidden_channels = 16
output_channels = 2
num_epochs = 200
learning_rate = 0.01

# Create and initialize the network
G = create_social_network(num_agents, avg_connections)
G, initial_informed_agents = initialize_agents(G, initial_informed)

# Simulate the initial diffusion process
informed_counts, initial_states_over_time, abm_new_informed = simulate_diffusion(G, infection_prob, steps)

# Prepare data for GNN
node_features, edge_index = prepare_gnn_data(G, initial_states_over_time)

# Convert to PyTorch geometric data format
train_data_list = []
for i in range(len(node_features) - 1):
    x = torch.tensor(node_features[i], dtype=torch.float)
    y = torch.tensor(node_features[i + 1], dtype=torch.float)
    train_data_list.append(Data(x=x, edge_index=edge_index))

# Initialize GNN model, optimizer, and loss criterion
model = GNN(num_node_features=2, hidden_channels=hidden_channels, output_channels=output_channels)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Train GNN model
for data in train_data_list:
    train_gnn(model, optimizer, criterion, data, data.y, num_epochs)

# Use the trained model to predict the next states
test_data = Data(x=torch.tensor(node_features[-1], dtype=torch.float), edge_index=edge_index)
predicted_states = predict_gnn(model, test_data)

# Convert predictions to informed counts
predicted_counts = [int(predicted_state[0].item() > 0.5) for predicted_state in predicted_states]
print("Predicted counts: ", predicted_counts)

# Visualization
# Function to visualize the diffusion process
def visualize_diffusion(G, states_over_time, predicted_states, steps, n_future_steps):
    fig, ax = plt.subplots(figsize=(7, 7))
    pos = nx.spring_layout(G)

    def update(i):
        ax.clear()
        color_map = {'unaware': 'red', 'informed': 'green'}
        if i < steps:
            colors = [color_map[state] for state in states_over_time[i]]
            ax.set_title(f"Actual Diffusion - Step {i+1}")
        else:
            future_index = i - steps
            colors = [color_map['unaware'] if state == 'unaware' else color_map['informed'] for state in predicted_states[future_index]]
            ax.set_title(f"Predicted Diffusion - Step {i+1}")
        nx.draw(G, pos, node_color=colors, with_labels=True, ax=ax)

    ani = FuncAnimation(fig, update, frames=steps + n_future_steps, repeat=False)
    plt.show()

# Generate predicted states over future steps
predicted_states_over_time = [initial_states_over_time[-1]]
current_state = torch.tensor(node_features[-1], dtype=torch.float)
for _ in range(n_future_steps):
    test_data = Data(x=current_state, edge_index=edge_index)
    predicted_state = predict_gnn(model, test_data)
    predicted_counts.append(sum(1 if state[0] > 0.5 else 0 for state in predicted_state))
    predicted_states = ['informed' if state[0] > 0.5 else 'unaware' for state in predicted_state]
    predicted_states_over_time.append(predicted_states)
    current_state = torch.tensor([[1, 0] if state == 'informed' else [0, 1] for state in predicted_states], dtype=torch.float)

# Visualize the diffusion process
visualize_diffusion(G, initial_states_over_time, predicted_states_over_time, steps, n_future_steps)

# Plot the comparison of informed counts
plt.figure()
plt.plot(range(steps), informed_counts, label="Actual Diffusion")
plt.plot(range(steps, steps + n_future_steps), predicted_counts[steps:], label="Predicted Diffusion", linestyle='dashed')
plt.xlabel('Time Steps')
plt.ylabel('Number of Informed Agents')
plt.legend()
plt.title('Comparison of Actual and Predicted Information Diffusion')
plt.show()
