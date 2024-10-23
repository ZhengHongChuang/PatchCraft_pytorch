from PIL import Image
import cv2
import numpy as np
import random
import concurrent.futures


def img_to_patches(img):
    # img = Image.open(fp=input_path)
    # if input_path[-3:] != 'jpg' and input_path[-4:] != 'jpeg':
    #     img = img.convert('RGB')
    # if img.size != (512, 512):
    #     img = img.resize(size=(512, 512))
    patch_size = 32
    grayscale_imgs = []
    imgs = []
    for i in range(0, img.height, patch_size):
        for j in range(0, img.width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            img_color = np.asarray(img.crop(box))
            grayscale_image = cv2.cvtColor(src=img_color, code=cv2.COLOR_RGB2GRAY)
            grayscale_imgs.append(grayscale_image.astype(dtype=np.int32))
            imgs.append(img_color)
    return grayscale_imgs, imgs

def get_pixel_var_degree_for_patch(patch) :
    x, y = patch.shape
    l1 = l2 = l3 = l4 = 0
    for i in range(y - 1):
        for j in range(x):
            l1 += abs(patch[j][i] - patch[j][i + 1])

    for i in range(y):
        for j in range(x - 1):
            l2 += abs(patch[j][i] - patch[j + 1][i])

    for i in range(y - 1):
        for j in range(x - 1):
            l3 += abs(patch[j][i] - patch[j + 1][i + 1])
            l4 += abs(patch[j + 1][i] - patch[j][i + 1])

    return l1 + l2 + l3 + l4


def extract_rich_and_poor_textures(variance_values, patches):
    combined = list(zip(variance_values, patches))
    combined.sort(key=lambda x: x[0], reverse=False)
    threshold = np.mean(variance_values)
    rich_texture_patches = []
    poor_texture_patches = []
    for var_value, patch in combined:
        if var_value <= threshold:
            poor_texture_patches.append(patch)

        else:
            rich_texture_patches.append(patch)


    return rich_texture_patches, poor_texture_patches



def get_complete_image(patches, coloured=True, reverse=False):
    if reverse:
        patches = patches[::-1]
    p_len = len(patches)
    if p_len > 64:
        patches = patches[0:64]
    while len(patches) < 64:
        patches.append(patches[random.randint(0, p_len - 1)])
    if coloured:
        grid = np.asarray(patches).reshape((8, 8, 32, 32, 3))
    else:
        grid = np.asarray(patches).reshape((8, 8, 32, 32))
    rows = [np.concatenate(grid[i, :], axis=1) for i in range(8)]
    img = np.concatenate(rows, axis=0)

    return img



def smash_n_reconstruct(img, coloured=True):
    gray_scale_patches, color_patches = img_to_patches(img)
    pixel_var_degree = []
    for patch in gray_scale_patches:
        pixel_var_degree.append(get_pixel_var_degree_for_patch(patch))
    if coloured:
        r_patch, p_patch = extract_rich_and_poor_textures(variance_values=pixel_var_degree, patches=color_patches)
    else:
        r_patch, p_patch = extract_rich_and_poor_textures(variance_values=pixel_var_degree, patches=gray_scale_patches)

    rich_texture, poor_texture = None, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        rich_texture_future = executor.submit(get_complete_image, r_patch, coloured,True)
        poor_texture_future = executor.submit(get_complete_image, p_patch, coloured,False)
        rich_texture = rich_texture_future.result()
        poor_texture = poor_texture_future.result()

    return rich_texture, poor_texture
# if __name__ == "__main__":
#     input_path = "/home/ubuntu/train/PatchCraft_pytorch/image.png"  # 输入图像路径

#     rich_texture_image, poor_texture_image = smash_n_reconstruct(input_path)
#     Image.fromarray(rich_texture_image).save("res/rich_1.jpg")  # 保存富有纹理的图像
#     Image.fromarray(poor_texture_image).save("res/poor_1.jpg")  # 保存贫乏纹理的图像
