from PIL import Image
import cv2
import numpy as np
import random
import concurrent.futures
# from filters import apply_filters

def img_to_patches(img) -> tuple:
    # img = Image.open(fp=input_path)
    # if input_path[-3:] != 'jpg' and input_path[-4:] != 'jpeg':
    #     img = img.convert('RGB')
    # if img.size != (256, 256):
    #     img = img.resize(size=(256, 256))
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
    """
    计算给定图像块的像素变异度
    ---------------------------------------
    ## 参数:
    - patch: 接受图像块的numpy数组格式
    """
    x, y = patch.shape
    l1 = l2 = l3 = l4 = 0

    # 计算 l1、l2、l3 和 l4
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
    """
    返回富有纹理和贫乏纹理的图像块
    --------------------------------------------------------------------
    ## 参数:
    - variance_values: 图像块的像素方差值列表
    - patches: 目标图像的彩色块
    """
    threshold = np.mean(variance_values)
    rich_texture_patches = []
    poor_texture_patches = []
    for i, j in enumerate(variance_values):
        if j >= threshold:
            rich_texture_patches.append(patches[i])
        else:
            poor_texture_patches.append(patches[i])
    
    return rich_texture_patches, poor_texture_patches

def get_complete_image(patches: list, coloured=True):
    """
    从富有或贫乏纹理的图像块开发完整的256x256图像
    ------------------------------------------------------------------
    ## 参数:
    - patches: 接受富有或贫乏纹理的图像块列表
    """
    random.shuffle(patches)
    p_len = len(patches)
    while len(patches) < 64:
        patches.append(patches[random.randint(0, p_len - 1)])
    
    if coloured:
        grid = np.asarray(patches).reshape((8, 8, 32, 32, 3))
    else:
        grid = np.asarray(patches).reshape((8, 8, 32, 32))

    # 连接列以保留行
    rows = [np.concatenate(grid[i, :], axis=1) for i in range(8)]

    # 连接行以创建最终图像
    img = np.concatenate(rows, axis=0)

    return img

def smash_n_reconstruct(img, coloured=True):
    """
    执行预处理的 SmashnReconstruct 部分
    返回富有纹理和贫乏纹理的图像块
    ----------------------------------------------------
    ## 参数:
    - input_path: 接受输入图像的路径
    """
    gray_scale_patches, color_patches = img_to_patches(img)
    pixel_var_degree = []
    for patch in gray_scale_patches:
        pixel_var_degree.append(get_pixel_var_degree_for_patch(patch))
    
    # r_patch = 富有纹理的图像块列表, p_patch = 贫乏纹理的图像块列表
    if coloured:
        r_patch, p_patch = extract_rich_and_poor_textures(variance_values=pixel_var_degree, patches=color_patches)
    else:
        r_patch, p_patch = extract_rich_and_poor_textures(variance_values=pixel_var_degree, patches=gray_scale_patches)

    rich_texture, poor_texture = None, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        rich_texture_future = executor.submit(get_complete_image, r_patch, coloured)
        poor_texture_future = executor.submit(get_complete_image, p_patch, coloured)

        rich_texture = rich_texture_future.result()
        poor_texture = poor_texture_future.result()

    return rich_texture, poor_texture
if __name__ == "__main__": 
    input_path = "imgs/fake/SFHQ_pt4_00001203.jpg"  # 输入图像路径
#     import os
#     base_name = os.path.basename(input_path)
    rich_texture_image, poor_texture_image = smash_n_reconstruct(input_path) 
    
    rich_texture,pool_texture = apply_filters(rich_texture_image, poor_texture_image)
    print(type(rich_texture_image))
#     Image.fromarray(rich_texture).save(f"res/fake/rich_texture1{base_name}.jpg")  # 保存富有纹理的图像
#     Image.fromarray(pool_texture).save(f"res/fake/poor_texture1{base_name}.jpg")  # 保存贫乏纹理的图像


    # poor_texture = apply_all_filters(poor_texture_image)
    # print(rich_texture_image, poor_texture_image)
    # Image.fromarray(rich_texture).save("res/rich_texture111221.jpg")  # 保存富有纹理的图像
    # Image.fromarray(poor_texture_image).save("res/poor_texture1.jpg")  # 保存贫乏纹理的图像