
import torch
from model import MTRT

def run_pipeline():
    data = torch.load('data/sample_eeg_motion_data.pt', weights_only=True)
    eeg, gt = data['eeg'], data['gt']

    model = MTRT()
    out = model(eeg)

    print("EEG Input Shape:", eeg.shape)
    print("Ground Truth Shape:", gt.shape)
    print("Predicted Output Shape:", out.shape)

if __name__ == "__main__":
    run_pipeline()
