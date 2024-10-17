
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
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0])
])


class TrainDataset(Dataset):

    def __init__(self,faceroot):
        self.data_list = []
        for dir in os.listdir(faceroot):
            if dir.startswith('0_real'):
                for img_dir in os.listdir(os.path.join(faceroot,dir)):
                    label=0
                    self.data_list.append({"image_path": os.path.join(faceroot,dir,img_dir), "label": label})
            elif dir.startswith('1_fake'):
                for img_dir in os.listdir(os.path.join(faceroot,dir)):
                    label=1
                    self.data_list.append({"image_path": os.path.join(faceroot,dir,img_dir), "label": label})
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
    
if __name__ == "__main__":
    # root = "imgs/fake/SFHQ_pt4_00001203.jpg"
    root = "/home/ubuntu/datasets/facedatasets"
    dataset = TrainDataset(root)
    rinch_txzture,pool_texture,_ =  dataset.__getitem__(102)
    # 将张量转换为 numpy.ndarray 
    rich_texture_img = rinch_txzture.squeeze(0).detach().numpy().astype('uint8')    
    poor_texture_img = pool_texture.squeeze(0).detach().numpy().astype('uint8')

    Image.fromarray(rich_texture_img).save("rich_texture.jpg")  # 保存富有纹理的图像    
    Image.fromarray(poor_texture_img).save("poor_texture.jpg")  # 保存贫乏纹理的图像
   

