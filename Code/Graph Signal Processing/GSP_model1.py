import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Create a simple graph
G = nx.karate_club_graph()

# Draw the graph
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title("Karate Club Graph")
plt.show()

# Generate a random signal on the graph
np.random.seed(0)
signal = np.random.randn(len(G.nodes))
print("Original Signal:", signal)

# Plot the signal
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color=signal, node_size=500, cmap=plt.cm.viridis)
plt.title("Graph Signal")
plt.show()

# Compute the graph Laplacian
L = nx.laplacian_matrix(G).toarray()

# Compute the eigendecomposition of the Laplacian
eigvals, eigvecs = eigh(L)
print("Eigenvalues:", eigvals)

# Compute the Graph Fourier Transform (GFT) of the signal
gft_signal = np.dot(eigvecs.T, signal)
print("Graph Fourier Transform of the Signal:", gft_signal)

# Plot the original signal and its GFT
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
nx.draw(G, with_labels=True, node_color=signal, node_size=500, cmap=plt.cm.viridis)
plt.title("Original Signal")

plt.subplot(1, 2, 2)
plt.plot(eigvals, gft_signal, 'o-')
plt.xlabel('Eigenvalue Index')
plt.ylabel('GFT Coefficient')
plt.title("Graph Fourier Transform")

plt.tight_layout()
plt.show()
