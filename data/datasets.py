import os
from torch.utils.data import Dataset
from torchvision import transforms
from preprocessing import smash_n_reconstruction, filters
import torch
from kornia import augmentation as K
from PIL import Image

Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.25, 0.5), p=0.1),
    K.RandomJPEG(jpeg_quality=(70, 100), p=0.1))
before_train  = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0])
])


class TrainDataset(Dataset):

    def __init__(self, faceroot, is_train=True):
        if is_train:
            faceroot = os.path.join(faceroot, 'train')
        else:
            faceroot = os.path.join(faceroot, 'val')

        self.data_list = []
        for dir in os.listdir(faceroot):
            if dir.startswith('nature') or dir.startswith('ai'):
                for img_dir in os.listdir(os.path.join(faceroot, dir)):
                    img_path = os.path.join(faceroot, dir, img_dir)
                    label = 0 if dir.startswith('nature') else 1
                    self.data_list.append({
                        "image_path": img_path,
                        "label": label
                    })

        self.transforms = before_train
        self.to_pil = transforms.ToPILImage()
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img = Image.open(self.data_list[index]["image_path"]).convert('RGB')
        img = self.transforms(img)
        img = self.to_pil(img)
        rich_patches, poor_patches = smash_n_reconstruction.smash_n_reconstruct(img)
        rich_texture,pool_texture = filters.apply_filters(rich_patches, poor_patches)
        rich_texture, pool_texture, = torch.from_numpy(rich_texture).to(dtype=torch.float64).unsqueeze(0), torch.from_numpy(pool_texture).to(dtype=torch.float64).unsqueeze(0)
        return rich_texture, pool_texture, self.data_list[index]["label"]
