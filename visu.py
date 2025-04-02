import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model import MTRT
import json
import matplotlib.animation as animation

def generate_trajectory_gif(pred, gt, sample_idx=0, filename="trajectory_animation.gif"):
    """
    Generate a GIF animation of joint trajectories over time.
    """
    pred = pred[sample_idx].reshape(-1, 6, 3).detach().cpu().numpy()
    gt = gt[sample_idx].reshape(-1, 6, 3).detach().cpu().numpy()

    joint_names = ['L-Shoulder', 'L-Elbow', 'L-Wrist', 'R-Shoulder', 'R-Elbow', 'R-Wrist']
    connections = [
        (0, 1),  # L-Shoulder to L-Elbow
        (1, 2),  # L-Elbow to L-Wrist
        (3, 4),  # R-Shoulder to R-Elbow
        (4, 5)   # R-Elbow to R-Wrist
    ]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([gt[:, :, 0].min(), gt[:, :, 0].max()])
    ax.set_ylim([gt[:, :, 1].min(), gt[:, :, 1].max()])
    ax.set_zlim([gt[:, :, 2].min(), gt[:, :, 2].max()])
    ax.set_title("Joint Trajectory Animation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Initialize plots for ground truth and predictions
    gt_lines = [ax.plot([], [], [], 'b--', label=f"GT {joint_names[start]} to {joint_names[end]}")[0] for start, end in connections]
    pred_lines = [ax.plot([], [], [], 'r-', label=f"Pred {joint_names[start]} to {joint_names[end]}")[0] for start, end in connections]

    def update(frame):
        """
        Update function for each frame in the animation.
        """
        for i, (start, end) in enumerate(connections):
            # Update ground truth lines
            gt_lines[i].set_data(gt[frame, [start, end], 0], gt[frame, [start, end], 1])
            gt_lines[i].set_3d_properties(gt[frame, [start, end], 2])

            # Update predicted lines
            pred_lines[i].set_data(pred[frame, [start, end], 0], pred[frame, [start, end], 1])
            pred_lines[i].set_3d_properties(pred[frame, [start, end], 2])

        return gt_lines + pred_lines

    ani = animation.FuncAnimation(fig, update, frames=gt.shape[0], interval=500, blit=True)
    ani.save(filename, writer='pillow')
    print(f"GIF animation saved to {filename}")

def save_trajectory(pred, gt, sample_idx=0, filename="trajectory_data.json"):
    """
    Save the predicted and ground truth trajectories to a JSON file.
    """
    pred = pred[sample_idx].reshape(-1, 6, 3).detach().cpu().numpy().tolist()
    gt = gt[sample_idx].reshape(-1, 6, 3).detach().cpu().numpy().tolist()

    data = {
        "pred": pred,
        "gt": gt,
        "joint_names": ['L-Shoulder', 'L-Elbow', 'L-Wrist', 'R-Shoulder', 'R-Elbow', 'R-Wrist']
    }

    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Trajectory data saved to {filename}")

import plotly.graph_objects as go

def visualize_trajectory_interactive(pred, gt, sample_idx=0):
    """
    Visualize trajectories interactively using Plotly.
    """
    pred = pred[sample_idx].reshape(-1, 6, 3).detach().cpu().numpy()
    gt = gt[sample_idx].reshape(-1, 6, 3).detach().cpu().numpy()

    joint_names = ['L-Shoulder', 'L-Elbow', 'L-Wrist', 'R-Shoulder', 'R-Elbow', 'R-Wrist']
    connections = [
        (0, 1),  # L-Shoulder to L-Elbow
        (1, 2),  # L-Elbow to L-Wrist
        (3, 4),  # R-Shoulder to R-Elbow
        (4, 5)   # R-Elbow to R-Wrist
    ]

    fig = go.Figure()

    # Add ground truth joints and connections
    for conn in connections:
        start, end = conn
        fig.add_trace(go.Scatter3d(
            x=gt[:, start, 0], y=gt[:, start, 1], z=gt[:, start, 2],
            mode='lines+markers',
            name=f"GT {joint_names[start]} to {joint_names[end]}",
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))

    # Add predicted joints and connections
    for conn in connections:
        start, end = conn
        fig.add_trace(go.Scatter3d(
            x=pred[:, start, 0], y=pred[:, start, 1], z=pred[:, start, 2],
            mode='lines+markers',
            name=f"Pred {joint_names[start]} to {joint_names[end]}",
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ))

    fig.update_layout(
        title="Interactive Joint Trajectory Reconstruction",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    fig.show()

    
def visualize_trajectory(pred, gt, sample_idx=0):
    """
    pred, gt: [B, T, 18]
    """
    pred = pred[sample_idx].reshape(-1, 6, 3).detach().numpy()  # Detach and convert to NumPy
    gt = gt[sample_idx].reshape(-1, 6, 3).detach().numpy()  # Detach and convert to NumPy

    joint_names = ['L-Shoulder', 'L-Elbow', 'L-Wrist', 'R-Shoulder', 'R-Elbow', 'R-Wrist']
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    for j in range(6):
        ax.plot(gt[:, j, 0], gt[:, j, 1], gt[:, j, 2], label=f'{joint_names[j]} GT', linestyle='--')
        ax.plot(pred[:, j, 0], pred[:, j, 1], pred[:, j, 2], label=f'{joint_names[j]} Pred', alpha=0.8)

    ax.set_title("Joint Trajectory Reconstruction")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.savefig('trajectory_visualization.png')

def run_pipeline():
    data = torch.load('data/sample_eeg_motion_data.pt')
    eeg, gt = data['eeg'], data['gt']

    model = MTRT()
    pred = model(eeg)

    print("EEG Input Shape:", eeg.shape)
    print("Ground Truth Shape:", gt.shape)
    print("Predicted Output Shape:", pred.shape)

    save_trajectory(pred, gt, sample_idx=0)  # Save trajectory data
    visualize_trajectory(pred, gt, sample_idx=0)
    visualize_trajectory_interactive(pred, gt, sample_idx=0)  # Interactive visualization
    generate_trajectory_gif(pred, gt, sample_idx=0, filename="trajectory_animation.gif")  # Generate GIF

if __name__ == "__main__":
    run_pipeline()