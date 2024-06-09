import os
import lightning as L
from dataloader import AerialImageDataset
from train import UNet
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch

train_path = "/teamspace/studios/this_studio/Aerial-Segmentation/train"
val_path = "/teamspace/studios/this_studio/Aerial-Segmentation/val"

data_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()]
    )

train_dataset = AerialImageDataset(os.path.join(train_path, 'images'), os.path.join(train_path, 'masks'), transform=data_transform)
val_dataset = AerialImageDataset(os.path.join(val_path, 'images'), os.path.join(val_path, 'masks'), transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = UNet(n_channels=3, n_classes=6)

trainer = L.Trainer(max_epochs=100)
trainer.fit(model, train_loader, val_loader)

torch.save(model.state_dict(), "/teamspace/studios/this_studio/Aerial-Segmentation/model.pth")
