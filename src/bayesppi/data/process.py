import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

# Create figure
fig, ax = plt.subplots(figsize=(12, 3))

# Define layers with positions
layers = [
    ("Input\n1×64×64×64", 0),
    ("Conv3D\n32×62×62×62", 2),
    ("MaxPool\n32×31×31×31", 4),
    ("Conv3D\n64×29×29×29", 6),
    ("MaxPool\n64×14×14×14", 8),
    ("Flatten", 10),
    ("Dense\n128", 12),
    ("Sigmoid\n1", 14)
]

y = 0.5
width = 1.5
height = 0.7

# Draw layers and arrows
for idx, (label, x) in enumerate(layers):
    ax.add_patch(Rectangle((x, y), width, height, fill=True, edgecolor='black', facecolor='lightgrey'))
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=10)
    # Draw arrow to next layer
    if idx < len(layers) - 1:
        ax.add_patch(FancyArrow(x + width, y + height/2, 0.5, 0, width=0.05, length_includes_head=True))

# Final formatting
ax.set_xlim(-0.5, 16)
ax.set_ylim(0, 2)
ax.axis('off')
plt.tight_layout()
plt.show()
