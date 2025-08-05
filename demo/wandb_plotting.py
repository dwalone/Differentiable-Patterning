import wandb
import matplotlib.pyplot as plt

api = wandb.Api()

# Project and run IDs
project_path = "s2629686-the-university-of-edinburgh/NCA"
run1 = api.run(f"{project_path}/db3pji4x")  # LR = 5e-5
run2 = api.run(f"{project_path}/jo3dbjmv")  # LR = 5e-4

# Fetch loss history
history1 = run1.history(keys=["Train/mean_loss"])
history2 = run2.history(keys=["Train/mean_loss"])

# Plotting
plt.figure(figsize=(7, 5))
plt.plot(history1["_step"], history1["Train/mean_loss"], label="TP = 0.0")
plt.plot(history2["_step"], history2["Train/mean_loss"], label="TP = 1.0")

plt.xlabel("Training Step")
plt.ylabel("Training Loss")
plt.title("Training Loss comparison")
plt.xlim(0, 25000)
plt.ylim(0, 0.05)
plt.legend()
plt.box(True)        # Keeps the outer box
plt.grid(False)      # Disables inner grid lines

# Save the figure
plt.tight_layout()
plt.savefig("st_training_loss_comparison.png", dpi=300)
print("Plot saved as training_loss_comparison.png")

