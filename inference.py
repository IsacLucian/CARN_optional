from multiprocessing import freeze_support
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from main import *

@timed
@torch.no_grad()
def transform_dataset_with_model(dataset, model, batch_size, device, num_workers=0, shuffle=False, drop_last=False):
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        drop_last=drop_last,
    )

    for (images,) in dataloader:
        images = images.to(device, non_blocking=True)
        _ = model(images)


def load_model(path, device):
    model = TransformNet()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


def test_inference_time(model):
    test_dataset = CustomDataset(data_path=DATA_PATH, train=False, cache=False)
    test_dataset = torch.stack(test_dataset.images)
    test_dataset = TensorDataset(test_dataset)

    t1 = transform_dataset_with_transforms(test_dataset)
    print(f"Sequential transforming each image took: {t1} on CPU. \n")

    bs = [16, 32, 64, 128, 256, 512]
    device = [torch.device('cuda'), torch.device('cpu')]
    shufle = [True, False]
    drop_last = [True, False]
    num_workers = [0, 2, 4]
    for d in device:
        for s in shufle:
            for n in drop_last:
                for b in bs:
                    for w in num_workers:
                        model.to(d)
                        t2 = transform_dataset_with_model(test_dataset, model, batch_size=b, device=d, num_workers=w, shuffle=s, drop_last=n)
                        print(f"Model time on {d.type} with batch_size={b}, num_workers={w}, shuffle={s}, drop_last={n}: {t2:.4f}s")


if __name__ == '__main__':
    freeze_support()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(SAVE_PATH, device=device)

    test_inference_time(model)
