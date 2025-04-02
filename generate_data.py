import torch

eeg = torch.randn(2, 1000, 50) # 2 samples, 1000 frames, 50 channels

gt = torch.randn(2, 60, 18) # # 2samples, 60 frames, 18 joints

"""
[
  0: 左肩 Left Shoulder,
  1: 左肘 Left Elbow,
  2: 左腕 Left Wrist,
  3: 右肩 Right Shoulder,
  4: 右肘 Right Elbow,
  5: 右腕 Right Wrist
]
"""

torch.save({'eeg': eeg, 'gt': gt}, 'data/sample_eeg_motion_data.pt')