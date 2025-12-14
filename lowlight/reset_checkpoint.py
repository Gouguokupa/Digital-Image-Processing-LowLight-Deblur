"""
Reset the best_val_loss in checkpoint to allow 256×256 training to save
"""

import torch

checkpoint_path = './checkpoints/best_model.pth'

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print(f"Current checkpoint info:")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Current val_loss: {checkpoint['val_loss']:.4f}")
print(f"  Best val_loss: {checkpoint['best_val_loss']:.4f}")

# Reset best_val_loss to a high value
checkpoint['best_val_loss'] = 999.0  # Set very high so any improvement will save

print(f"\nResetting best_val_loss to 999.0")
print(f"This allows your 256×256 training to save as 'best'")

# Save modified checkpoint
torch.save(checkpoint, checkpoint_path)

print(f"\n✅ Checkpoint updated!")
print(f"Now resume training and it will save improvements")

