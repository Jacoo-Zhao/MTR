# 🧠 MTRT: Motion Trajectory Reconstruction Transformer

This repository implements the **MTRT** model for reconstructing 3D upper-limb motion trajectories from EEG signals, inspired by the paper:

> **MTRT: Motion Trajectory Reconstruction Transformer for EEG-Based BCI Decoding**  
> (TNSRE 2024)

---

## 📦 Project Structure

```
MTRT/
├── data/                         # Sample EEG-motion data
│   └── sample_eeg_motion_data.pt
├── utils/
│   ├── loss.py                  # Geometric constraint losses
│   └── ...
├── model.py                     # Core model: Encoder + Decoder
├── pipeline.py                  # Inference demo
├── train.py                     # Training entry
├── visu.py                      # 3D trajectory visualization
├── trajectory_visualization.png # Sample output (PNG)
├── trajectory_animation.gif     # Sample animation (GIF)
├── trajectory_data.json         # Optional exported data
└── README.md
```

---

## 🚀 How It Works

- **Input**: Pre-processed EEG sequence `[B, T=1000, C=50]`
- **Output**: 3D joint trajectory sequence `[B, T=60, D=18]`
  - 6 joints × 3D (L-shoulder, L-elbow, L-wrist, R-shoulder, R-elbow, R-wrist)

- **Model**: Transformer-based
  - **6-layer EEG Encoder** with self-attention
  - **6-layer Trajectory Decoder** with cross-attention
  - Positional encoding added to both streams

- **Loss Function**: Combines MSE and geometric constraints
  - e.g., limb length consistency, left-right symmetry

---

## 📊 Visualization

- Run `visu.py` to visualize predicted and ground-truth trajectories in 3D:

```bash
python visu.py
```

- Outputs: `trajectory_visualization.png`, `trajectory_animation.gif`

---

## 🔁 Planned Feature: Autoregressive Inference (TBD)

This repo **currently supports parallel decoding (non-autoregressive)**, where the decoder receives a full sequence of zeros as input and predicts the entire trajectory at once.

> ✅ A future update will **add support for autoregressive trajectory generation** (step-by-step decoding), which is critical for:
>
> - Real-time applications (e.g., robotic control)
> - More accurate temporal modeling
> - Online EEG-to-motion systems

Stay tuned.

---

## 🛠️ Quick Start

```bash
# Install dependencies
pip install torch matplotlib

# Run inference on sample data
python pipeline.py

# Run training (tbd)
python train.py

# Visu
python visu.py
```

---

## 📌 Citation

If you use this code or data, please cite the original paper (TBD).

---

## 👤 Author

Maintained by [@ziyzhao](mailto:ziyi.zhao-2@student.uts.edu.au)  
HAILab @ UTS