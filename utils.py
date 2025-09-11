from typing import Optional, Tuple, Any
import os
import math
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.utils import save_image


def save_all_images(
        imgs: Any,
        labels: Any,
        output_path: str,
        start_idx: Optional[int] = 0,
        ) -> None:
    amount = imgs.shape[0]
    for i in range(amount):
        save_image(imgs[i], f'{output_path}/%05d_%d.png'%(i+start_idx,labels[i]))

def load_samples(
    path: str,
    start_idx: Optional[int] = 0,
    end_idx: Optional[int] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Any, Any]:   

    images, labels = [], []
    basepath = r""
    samplepath = os.path.join(basepath, f"{path}")
    files = os.listdir(path)
    if end_idx is None:
        end_idx = len(files)

    for i in range(start_idx,end_idx):
        # get filename and label
        file = [n for n in files if f"{i:05d}_" in n][0]
        label = int(file.split(".")[0].split("_")[-1])

        # open file
        path = os.path.join(samplepath, file)
        image = Image.open(path)
        
        if shape is not None:
            image = image.resize(shape)
        
        image = np.asarray(image, dtype=np.float32)

        if image.ndim == 2:
            image = image[..., np.newaxis]

        assert image.ndim == 3
        
        image = np.transpose(image, (2, 0, 1))
        
        images.append(image)
        labels.append(label)
    
    images_ = np.stack(images)
    labels_ = np.array(labels)

    images_ = images_ / 255
    images_ = np.ascontiguousarray(images_)
    
    return images_, labels_

def load_image_tensor(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0

    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

def save_image_tensor(x, path):
    arr = (x.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def predict(
    model,
    imgs: Tensor,
    batch_size: int=100,
    )-> Tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    count = len(imgs)
    batches = math.ceil(count/batch_size)
    logits = Tensor([])
    for b in range(batches):
        if b == batches-1:
            idx = range(b*batch_size,count)
        else:
            idx = range(b*batch_size,(b+1)*batch_size)
        with torch.no_grad():
            logits_batch = model.to(device)(imgs[idx].to(device)).cpu()
        logits = torch.cat((logits, logits_batch), 0)
    
    return logits