# evaluate.py
import torch
import matplotlib.pyplot as plt
from covid_classification import RadiographyCNN, run_epoch, evaluate_detailed, test_dl, DEVICE, best_model_path

# Load model
model = RadiographyCNN().to(DEVICE)
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

# Plot & print test metrics
test_loss, test_acc = run_epoch(model, test_dl, False)
print(f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
evaluate_detailed(model, test_dl)
