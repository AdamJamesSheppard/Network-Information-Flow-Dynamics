import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Drone:
    def __init__(self, id, position, velocity, information=False):
        self.id = id
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.information = information  # Information state: True if carrying message

    def update_position(self, dt, bounds):
        self.position += self.velocity * dt
        # Apply boundary conditions (reflective boundaries)
        for i in range(len(self.position)):
            if self.position[i] < bounds[0][i]:
                self.position[i] = bounds[0][i]
                self.velocity[i] = -self.velocity[i]
            elif self.position[i] > bounds[1][i]:
                self.position[i] = bounds[1][i]
                self.velocity[i] = -self.velocity[i]

    def correlated_random_walk(self, alpha, dt, max_speed):
        angle_change = np.random.uniform(-alpha, alpha)
        speed_change = np.random.uniform(-0.1, 0.1)
        angle = np.arctan2(self.velocity[1], self.velocity[0]) + angle_change
        speed = np.linalg.norm(self.velocity) + speed_change
        speed = max(0.1, min(speed, max_speed))  # Ensure speed is within bounds
        self.velocity = speed * np.array([np.cos(angle), np.sin(angle)])

    def send_message(self, source_pos, target_pos, message_radius):
        # Check if near the information source or the target
        if np.linalg.norm(self.position - source_pos) < message_radius:
            self.information = True
        if np.linalg.norm(self.position - target_pos) < message_radius and self.information:
            self.information = False  # Message delivered

def compute_laplacian(G):
    A = nx.adjacency_matrix(G).todense()
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return L

def animate(frame):
    global drones, G, bounds, ax1, ax2, ax3, sm, cbar, source_pos, target_pos

    ax1.clear()
    ax2.clear()
    ax3.clear()

    # Update positions and information states
    for drone in drones:
        drone.correlated_random_walk(alpha=0.2, dt=dt, max_speed=1.0)
        drone.update_position(dt, bounds)
        drone.send_message(source_pos, target_pos, message_radius=1.0)

    # Update graph for drone dynamics
    positions = {drone.id: drone.position for drone in drones}
    G.clear()
    G.add_nodes_from(positions.keys())
    for drone in drones:
        for other in drones:
            if drone.id != other.id and np.linalg.norm(drone.position - other.position) < 3.0:
                G.add_edge(drone.id, other.id)

    # Compute the Laplacian matrix
    L = compute_laplacian(G)

    # Draw drone dynamics
    node_colors = ['green' if drone.information else 'blue' for drone in drones]
    nx.draw(G, pos=positions, ax=ax1, node_color=node_colors, node_size=100, with_labels=False, edge_color='gray')
    ax1.scatter(*source_pos, color='red', s=200, marker='o', label='Source')
    ax1.scatter(*target_pos, color='black', s=200, marker='x', label='Target')
    ax1.set_xlim(bounds[0][0], bounds[1][0])
    ax1.set_ylim(bounds[0][1], bounds[1][1])
    ax1.set_title(f"Drone Swarm - Frame {frame}")
    ax1.legend()

    # Draw the Laplacian matrix
    ax2.matshow(L, cmap='viridis')
    ax2.set_title("Graph Laplacian")

    # Draw positions only
    for drone in drones:
        ax3.plot(drone.position[0], drone.position[1], 'bo' if not drone.information else 'go')
    ax3.scatter(*source_pos, color='red', s=200, marker='o', label='Source')
    ax3.scatter(*target_pos, color='black', s=200, marker='x', label='Target')
    ax3.set_xlim(bounds[0][0], bounds[1][0])
    ax3.set_ylim(bounds[0][1], bounds[1][1])
    ax3.set_title(f"Drone Positions - Frame {frame}")
    ax3.legend()

# Initialize parameters
num_drones = 10
time_steps = 200
dt = 0.1
bounds = [[0, 0], [10, 10]]  # [[x_min, y_min], [x_max, y_max]]
source_pos = np.array([9, 9])  # Information source position
target_pos = np.array([1, 1])  # Target position

# Create drones with initial random positions, zero velocities, and no initial information
drones = [Drone(id=i, position=np.random.rand(2) * 10, velocity=np.random.rand(2) * 2 - 1) for i in range(num_drones)]

# Create graph
G = nx.Graph()

# Create figure for animation
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))

ani = animation.FuncAnimation(fig, animate, frames=time_steps, interval=50)

# To display the animation inline (if in a Jupyter notebook)
# from IPython.display import HTML
# HTML(ani.to_jshtml())

# To save the animation as a file
ani.save('drone_swarm_simulation_laplacian.mp4', writer='ffmpeg')

plt.show()
