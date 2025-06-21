import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class SpamAccountTracker:
    def __init__(self, threshold=3):
        self.threshold = threshold  # Number of spam reports needed to mark as spam
        self.G = nx.Graph()
        self.account_reports = defaultdict(int)  # Tracks number of reports per account
        self.spam_accounts = set()  # Track confirmed spam accounts
        self.edge_history = []  # Track connections for visualization
        self.pos = None  # Store graph layout positions
        
    def add_interaction(self, account1, account2, is_spam_report=False):
        """Add an interaction between two accounts, optionally a spam report"""
        # Add nodes if they don't exist
        if account1 not in self.G:
            self.G.add_node(account1, spam=False)
        if account2 not in self.G:
            self.G.add_node(account2, spam=False)
            
        # Add edge between them
        self.G.add_edge(account1, account2)
        self.edge_history.append((account1, account2))
        
        # If this is a spam report, update counts
        if is_spam_report:
            self.account_reports[account2] += 1
            if self.account_reports[account2] >= self.threshold and account2 not in self.spam_accounts:
                self.spam_accounts.add(account2)
                self.G.nodes[account2]['spam'] = True
                
    def draw_graph(self):
        """Draw the current state of the graph"""
        plt.figure(figsize=(10, 8))
        
        # Use spring layout if we haven't calculated positions yet
        if self.pos is None or len(self.pos) != len(self.G.nodes()):
            self.pos = nx.spring_layout(self.G, k=0.15, iterations=50)
        
        # Color nodes based on spam status
        node_colors = ['red' if self.G.nodes[node]['spam'] else 'skyblue' for node in self.G.nodes()]
        node_sizes = [300 if self.G.nodes[node]['spam'] else 150 for node in self.G.nodes()]
        
        # Draw the graph
        nx.draw_networkx_nodes(self.G, self.pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(self.G, self.pos, alpha=0.5)
        nx.draw_networkx_labels(self.G, self.pos)
        
        # Add legend
        plt.scatter([], [], c='red', label='Spam Account')
        plt.scatter([], [], c='skyblue', label='Normal Account')
        plt.legend(scatterpoints=1, frameon=True)
        
        plt.title(f"Spam Account Network (Threshold: {self.threshold} reports)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    def animate_graph(self, frames=10):
        """Create an animation of graph growth"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update(frame):
            ax.clear()
            # Create a subgraph with the first 'frame' edges
            subgraph = nx.Graph()
            nodes = set()
            
            for i in range(min(frame, len(self.edge_history))):
                account1, account2 = self.edge_history[i]
                subgraph.add_edge(account1, account2)
                nodes.update([account1, account2])
            
            # Add node attributes
            for node in nodes:
                subgraph.add_node(node, spam=node in self.spam_accounts)
            
            # Only calculate layout if needed
            if len(subgraph.nodes()) > 0:
                pos = nx.spring_layout(subgraph, k=0.15, iterations=20)
                
                # Color nodes
                node_colors = ['red' if subgraph.nodes[node]['spam'] else 'skyblue' for node in subgraph.nodes()]
                node_sizes = [300 if subgraph.nodes[node]['spam'] else 150 for node in subgraph.nodes()]
                
                # Draw
                nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
                nx.draw_networkx_edges(subgraph, pos, alpha=0.5, ax=ax)
                nx.draw_networkx_labels(subgraph, pos, ax=ax)
                
                # Add legend
                ax.scatter([], [], c='red', label='Spam Account')
                ax.scatter([], [], c='skyblue', label='Normal Account')
                ax.legend(scatterpoints=1, frameon=True)
                
                ax.set_title(f"Spam Account Network Growth (Threshold: {self.threshold})\nStep {frame} of {frames}")
                ax.axis('off')
        
        ani = FuncAnimation(fig, update, frames=frames, interval=1000, repeat=False)
        plt.close()
        return HTML(ani.to_jshtml())

# Example usage
if __name__ == "__main__":
    # Create tracker with threshold of 3 reports
    tracker = SpamAccountTracker(threshold=3)
    
    # Generate some random interactions
    accounts = [f"User{i}" for i in range(1, 11)]
    interactions = []
    
    # Create some normal interactions
    for _ in range(20):
        a1, a2 = random.sample(accounts, 2)
        interactions.append((a1, a2, False))
    
    # Create some spam reports targeting certain accounts
    spam_targets = ["User2", "User5", "User7"]
    for target in spam_targets:
        for _ in range(3):  # Enough to reach threshold
            reporter = random.choice([a for a in accounts if a != target])
            interactions.append((reporter, target, True))
    
    # Shuffle interactions
    random.shuffle(interactions)
    
    # Process interactions
    for i, (a1, a2, is_spam) in enumerate(interactions):
        tracker.add_interaction(a1, a2, is_spam)
    
    # Visualize final graph
    tracker.draw_graph()
    
    # To see the animation (works best in Jupyter notebook)
    # Uncomment the following line if using Jupyter notebook
    # display(tracker.animate_graph(frames=len(interactions)))