import matplotlib.pyplot as plt
import numpy as np
import os

def create_neural_network_diagram():
    """Create a simple neural network diagram for the student handout."""
    # Set up the figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()
    ax.set_xlim([0, 4])
    ax.set_ylim([0, 4])
    
    # Turn off the axis
    plt.axis('off')
    
    # Define layers
    input_layer_nodes = 8
    hidden_layer_nodes = 6
    output_layer_nodes = 4
    
    # Colors
    input_color = '#3498db'  # Blue
    hidden_color = '#2ecc71'  # Green
    output_color = '#e74c3c'  # Red
    
    # Positions
    input_x = 0.5
    hidden_x = 2.0
    output_x = 3.5
    
    # Draw input layer
    input_positions = []
    for i in range(input_layer_nodes):
        y = 0.5 + (3.0 / (input_layer_nodes - 1)) * i
        input_positions.append((input_x, y))
        circle = plt.Circle((input_x, y), 0.15, color=input_color, fill=True)
        ax.add_patch(circle)
    
    # Draw hidden layer
    hidden_positions = []
    for i in range(hidden_layer_nodes):
        y = 1.0 + (2.0 / (hidden_layer_nodes - 1)) * i
        hidden_positions.append((hidden_x, y))
        circle = plt.Circle((hidden_x, y), 0.15, color=hidden_color, fill=True)
        ax.add_patch(circle)
    
    # Draw output layer
    output_positions = []
    output_labels = ['Happy', 'Sad', 'Surprised', 'Neutral']
    for i in range(output_layer_nodes):
        y = 1.0 + (2.0 / (output_layer_nodes - 1)) * i
        output_positions.append((output_x, y))
        circle = plt.Circle((output_x, y), 0.15, color=output_color, fill=True)
        ax.add_patch(circle)
        # Add labels for output nodes
        plt.text(output_x + 0.3, y, output_labels[i], fontsize=12, verticalalignment='center')
    
    # Draw connections between input and hidden layers
    for i, p1 in enumerate(input_positions):
        for j, p2 in enumerate(hidden_positions):
            # Draw with varying transparency to make it look less cluttered
            alpha = 0.2 if i % 2 == 0 and j % 2 == 0 else 0.1
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=alpha)
    
    # Draw connections between hidden and output layers
    for i, p1 in enumerate(hidden_positions):
        for j, p2 in enumerate(output_positions):
            # Draw with varying transparency
            alpha = 0.2 if i % 2 == 0 and j % 2 == 0 else 0.1
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', alpha=alpha)
    
    # Add layer labels
    plt.text(input_x, 4.0, 'Input Layer\n(Image Features)', fontsize=14, 
             horizontalalignment='center', verticalalignment='center')
    plt.text(hidden_x, 4.0, 'Hidden Layer', fontsize=14,
             horizontalalignment='center', verticalalignment='center')
    plt.text(output_x, 4.0, 'Output Layer\n(Emotions)', fontsize=14,
             horizontalalignment='center', verticalalignment='center')
    
    # Add input layer dots to indicate more nodes
    plt.text(input_x, 0.1, '...', fontsize=20, horizontalalignment='center')
    
    # Add title
    plt.text(2.0, 0.1, 'Neural Network for Emotion Recognition', fontsize=16,
             horizontalalignment='center', fontweight='bold')
    
    # Save the diagram
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    output_path = os.path.join(docs_dir, 'neural_network_diagram.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Neural network diagram created at: {output_path}")

if __name__ == "__main__":
    create_neural_network_diagram() 