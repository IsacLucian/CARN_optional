import gc
import os
from functools import wraps
from multiprocessing import freeze_support
from time import time

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = CURRENT_DIR + '/data'
OUT_DIR = CURRENT_DIR + '/outputs'
SAVE_PATH = CURRENT_DIR + '/transform_net.pt'

def timed(fn: callable):
    @wraps(fn)
    def wrap(*args, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time()
        fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time()
        return end - start

    return wrap


def get_cifar10_images(data_path, train):
    initial_transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    cifar_10_images = CIFAR10(root=data_path, train=train, transform=initial_transforms, download=True)
    return [image for image, label in cifar_10_images]


class CustomDataset(Dataset):
    def __init__(self, data_path, train, cache):
        self.images = get_cifar10_images(data_path, train)
        self.cache = cache
        self.transforms = v2.Compose([
            v2.Resize((28, 28), antialias=True),
            v2.Grayscale(),
            v2.functional.hflip,
            v2.functional.vflip,
        ])
        if cache:
            self.labels = [self.transforms(x) for x in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if self.cache:
            return self.images[i], self.labels[i]
        return self.images[i], self.transforms(self.images[i])


class TransformNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_gray = nn.Conv2d(3, 1, kernel_size=1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, size=(28, 28), mode="bilinear", align_corners=False, antialias=True)
        x = self.to_gray(x)
        x = torch.flip(x, dims=(-1,))
        x = torch.flip(x, dims=(-2,))

        return x.clamp(0.0, 1.0)


@timed
def transform_dataset_with_transforms(dataset):
    transforms = v2.Compose([
        v2.Resize((28, 28), antialias=True),
        v2.Grayscale(),
        v2.functional.hflip,
        v2.functional.vflip,
    ])

    for image in dataset.tensors[0]:
        transforms(image)


@timed
@torch.no_grad()
def transform_dataset_with_model(dataset, model, batch_size, device, num_workers=0):
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )

    for (images,) in dataloader:
        images = images.to(device, non_blocking=True)
        _ = model(images)


def train_transform_model(model, data_path, device, batch_size, lr, max_epochs, patience, min_delta, save_path):
    full = CustomDataset(data_path=data_path, train=True, cache=False)

    val_len = max(1, int(0.1 * len(full)))
    train_len = len(full) - val_len
    train_ds, val_ds = random_split(full, [train_len, val_len], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val = float("inf")
    bad_epochs = 0

    writer = SummaryWriter(log_dir=os.path.join(CURRENT_DIR, f'logs_{device}'))
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        t0 = time()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(x)

            loss = F.mse_loss(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * x.size(0)
            train_n += x.size(0)

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x)
                vloss = F.mse_loss(pred, y)
                val_loss_sum += float(vloss.item()) * x.size(0)
                val_n += x.size(0)

        train_loss = train_loss_sum / max(train_n, 1)
        val_loss = val_loss_sum / max(val_n, 1)
        writer.add_scalar('Time/Epoch', time() - t0, epoch)
        writer.add_scalar('Loss/Train_MSE', train_loss, epoch)
        writer.add_scalar('Loss/Val_MSE', val_loss, epoch)
        print(f"Epoch {epoch:02d} | train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f}")

        if val_loss < best_val - min_delta:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), save_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping (no val improvement for {patience} epochs). Best val: {best_val:.6f}")
                break
    
    writer.close()
    model.load_state_dict(torch.load(save_path, map_location=device))
    
    return model


@torch.no_grad()
def export_comparison_images(model, data_path, device, out_dir, n=5):
    os.makedirs(out_dir, exist_ok=True)
    ds = CustomDataset(data_path=data_path, train=False, cache=False)
    model.eval().to(device)

    for i in range(min(n, len(ds))):
        x, y = ds[i]
        pred = model(x.unsqueeze(0).to(device)).cpu().squeeze(0)

        save_image(y, os.path.join(out_dir, f"sample_{i}_gt.png"))
        save_image(pred, os.path.join(out_dir, f"sample_{i}_pred.png"))


def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = TransformNet()

    train_transform_model(
        model,
        device=device,
        max_epochs=30,
        patience=4,
        batch_size=256,
        lr=2e-3,
        min_delta=1e-4,
        save_path=SAVE_PATH,
        data_path=DATA_PATH,
    )

    export_comparison_images(
        model, 
        device=device, 
        data_path=DATA_PATH, 
        out_dir=OUT_DIR, 
        n=5
    )


if __name__ == '__main__':
    freeze_support()
    main()
