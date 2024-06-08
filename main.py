import os
import lightning as L
from dataloader import AerialImageDataset
from train import UNet
from torch.utils.data import DataLoader

train_path = "/teamspace/studios/this_studio/AerialSegmentation/train"
val_path = "/teamspace/studios/this_studio/AerialSegmentation/val"

data_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = AerialImageDataset(os.path.join(train_path, 'images'), os.path.join(train_path, 'masks'), transform=data_transform)
val_dataset = AerialImageDataset(os.path.join(val_path, 'images'), os.path.join(val_path, 'masks'), transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = UNet(n_channels=3, n_classes=6)

trainer = L.Trainer(max_epochs=100)
trainer.fit(model, train_loader, val_loader)
