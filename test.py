import numpy as np
import matplotlib.pyplot as plt

# Assuming Q is your input matrix with shape (16, 4)
# For demonstration, I'm initializing Q with random numbers.
# Replace this with your actual matrix.
Q = np.random.rand(16, 4)

# This function will update the subplot at location (1,2,2) with the new Q values
def update_subplot_122(Q, ax2):
    # Clear previous texts and lines
    ax2.clear()
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Reshape Q to a 4x4x4 tensor
    Q_reshaped = Q.reshape((4, 4, 4))
    
    # Define positions for the text within each cell: left, down, right, up
    positions = [(-0.2, 0.5), (0.5, -0.2), (1.2, 0.5), (0.5, 1.2)]
    
    # Iterate over the reshaped matrix to place text
    for i in range(4):
        for j in range(4):
            # Extract the 1x4 vector for the current cell
            vector = Q_reshaped[i, j]
            
            # Place each number in its position
            for k, (x_pos, y_pos) in enumerate(positions):
                ax2.text(j + x_pos, 3 - i + y_pos, f'{vector[k]:.2f}', ha='center', va='center',
                         fontsize=8, transform=ax2.transData)

    # Set limits and aspect
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(-0.5, 3.5)
    ax2.set_aspect('equal')
    
    # Draw grid lines
    for x in range(4):
        ax2.axhline(x - 0.5, lw=2, color='k', zorder=5)
        ax2.axvline(x - 0.5, lw=2, color='k', zorder=5)
    
    # Adjust layout and draw the canvas
    plt.draw()

# Initialize the figure with 1x2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Placeholder for the first subplot
ax1.text(0.5, 0.5, 'Placeholder for ax1', ha='center', va='center')
ax1.axis('off')

# Update the subplot (1,2,2) with the initial Q
update_subplot_122(Q, ax2)

# Show the initial plot
plt.show()

# Now, if you want to update the subplot continuously as Q changes,
# you would call the function `update_subplot_122` with new Q values
# inside a loop, followed by plt.pause or similar in a live script.
# Example:
# for _ in range(10):  # Loop to update the plot 10 times
#     Q = np.random.rand(16, 4)  # Generate new Q values
#     update_subplot_122(Q, ax2)  # Update the plot
#     plt.pause(1)  # Pause for a second to see the plot update
# Please note that plt.pause is for demonstration and will not work in this static notebook environment.
