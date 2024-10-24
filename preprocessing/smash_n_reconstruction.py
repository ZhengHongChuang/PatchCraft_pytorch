from PIL import Image
import cv2
import numpy as np
import random
import concurrent.futures


# def img_to_patches(img):
#     # img = Image.open(fp=input_path)
#     # if input_path[-3:] != 'jpg' and input_path[-4:] != 'jpeg':
#     #     img = img.convert('RGB')
#     # if img.size != (512, 512):
#     #     img = img.resize(size=(512, 512))
#     patch_size = 32
#     grayscale_imgs = []
#     imgs = []
#     for i in range(0, img.height, patch_size):
#         for j in range(0, img.width, patch_size):
#             box = (j, i, j + patch_size, i + patch_size)
#             img_color = np.asarray(img.crop(box))
#             grayscale_image = cv2.cvtColor(src=img_color, code=cv2.COLOR_RGB2GRAY)
#             grayscale_imgs.append(grayscale_image.astype(dtype=np.int32))
#             imgs.append(img_color)
#     return grayscale_imgs, imgs

def random_crop_patches(image_path, patch_size=(32, 32), num_patches=192):
    image = Image.open(image_path).convert('RGB')
    w, h = image.size
    patches = []
    gray_patches = []
    for _ in range(num_patches):
        x = np.random.randint(0, w - patch_size[0])
        y = np.random.randint(0, h - patch_size[1])
        patch = np.asarray(
            image.crop((x, y, x + patch_size[0], y + patch_size[1])))
        gray_patches.append(cv2.cvtColor(src=patch, code=cv2.COLOR_RGB2GRAY))
        patches.append(patch)
    return gray_patches, patches

def get_pixel_var_degree_for_patch(patch) :
    patch_copy = np.copy(patch).astype(np.float32)
    x, y= patch.shape
    l1 = l2 = l3 = l4 = 0
    for i in range(x):
        for j in range(y - 1):
            l1 += abs(patch_copy[i][j] - patch_copy[i][j + 1])
    for i in range(x - 1):
        for j in range(y):
            l2 += abs(patch_copy[i][j] - patch_copy[i + 1][j])

    for i in range(x - 1):
        for j in range(y - 1):
            l3 += abs(patch_copy[i][j] - patch_copy[i + 1][j + 1])
            l4 += abs(patch_copy[i + 1][j] - patch_copy[i][j + 1])
    return l1 + l2 + l3 + l4


def extract_rich_and_poor_textures(variance_values, patches):
    combined = list(zip(variance_values, patches))
    combined.sort(key=lambda x: x[0], reverse=False)
    poor_texture_patches = [patch for _, patch in combined[:64]]
    rich_texture_patches = [patch for _, patch in combined[-64:]]



    return rich_texture_patches, poor_texture_patches



def get_complete_image(patches, coloured=True, reverse=False):
    if reverse:
        patches = patches[::-1]

    if coloured:
        grid = np.asarray(patches).reshape((8, 8, 32, 32, 3))
    else:
        grid = np.asarray(patches).reshape((8, 8, 32, 32))

    rows = [np.concatenate(grid[i, :], axis=1) for i in range(8)]
    img = np.concatenate(rows, axis=0)
    return img


def smash_n_reconstruct(img, coloured=True):
    gray_scale_patches, color_patches = random_crop_patches(image_path=img)
    pixel_var_degree = []
    for patch in gray_scale_patches:
        pixel_var_degree.append(get_pixel_var_degree_for_patch(patch))
    if coloured:
        r_patch, p_patch = extract_rich_and_poor_textures(
            variance_values=pixel_var_degree, patches=color_patches)
    else:
        r_patch, p_patch = extract_rich_and_poor_textures(
            variance_values=pixel_var_degree, patches=gray_scale_patches)


    rich_texture, poor_texture = None, None
    rich_texture = get_complete_image(r_patch, coloured, True)
    poor_texture = get_complete_image(p_patch, coloured, False)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        rich_texture_future = executor.submit(get_complete_image, r_patch,
                                              coloured, True)
        poor_texture_future = executor.submit(get_complete_image, p_patch,
                                              coloured, False)
        rich_texture = rich_texture_future.result()
        poor_texture = poor_texture_future.result()

    return rich_texture, poor_texture


if __name__ == "__main__":
    input_path = "/home/ubuntu/train/PatchCraft_pytorch/datasets"
    output_path = "/home/ubuntu/train/PatchCraft_pytorch/s2r_res"
    import os
    from filters import apply_filters
    for dir in os.listdir(input_path):
        if dir.startswith("0_real") or dir.startswith("1_fake"):
            os.makedirs(os.path.join(output_path, dir), exist_ok=True)
            for file in os.listdir(os.path.join(input_path, dir)):
                rich_texture_image, poor_texture_image = smash_n_reconstruct(
                    os.path.join(input_path, dir, file))
                Image.fromarray(rich_texture_image).save(
                    os.path.join(output_path, dir, file[:-4] + "_rich.jpg"))
                Image.fromarray(poor_texture_image).save(
                    os.path.join(output_path, dir, file[:-4] + "_poor.jpg"))
                rich_texture,poor_texture = apply_filters(rich_texture_image,poor_texture_image)
                Image.fromarray(rich_texture).save(
                    os.path.join(output_path, dir, file[:-4] + "_rich_filtered.jpg"))
                Image.fromarray(poor_texture).save(
                    os.path.join(output_path, dir, file[:-4] + "_poor_filtered.jpg"))

    print("Done")
