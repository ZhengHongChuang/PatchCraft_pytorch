import numpy as np
from PIL import Image


def img_to_patches(img, patch_size=32, stride=36):
    """
    从512x512的图像中提取补丁
    """
    patches = []
    for i in range(0, img.height - patch_size + 1, stride):
        for j in range(0, img.width - patch_size + 1, stride):
            box = (j, i, j + patch_size, i + patch_size)
            img_patch = img.crop(box)
            patches.append(np.asarray(img_patch))

    return patches


# 示例
input_path = 'image.png'
img = Image.open(input_path).convert('RGB')
img = img.resize((512, 512))  # 确保图像是512x512

# 提取补丁
patches = img_to_patches(img, patch_size=32, stride=36)
print(f'提取的补丁数量: {len(patches)}')
