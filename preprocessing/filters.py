import numpy as np
import cv2
filter_a = [
    np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -1, 0, 0],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -1, 0, 0],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, -1, 1, 0],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 0, 0],
              [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 0, 0],
              [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 0, 0],
              [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -1, 0, 0],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -1, 0, 0],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
]
filter_b = [
    np.array([[0, 0, 0, 0, 0], [0, 2, 1, 0, 0], [0, 1, -3, 0, 0],
              [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-1, 3, -3, 1, 0],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, -3, 0, 0],
              [0, 2, 1, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -3, 0, 0],
              [0, 0, 3, 0, 0], [0, 0, -1, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -3, 1, 0],
              [0, 0, 1, 2, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -3, 3, -1],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 0, 1, 2, 0], [0, 0, -3, 1, 0],
              [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, -1, 0, 0], [0, 0, 3, 0, 0], [0, 0, -3, 0, 0],
              [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32)
]
filter_c = [
    np.array([[-1, 2, -2, 0, 0], [2, -6, 8, 0, 0], [-2, 8, -12, 0, 0],
              [2, -6, 8, 0, 0], [-1, 2, -2, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
             dtype=np.float32),
    np.array([[0, 0, -2, 2, -1], [0, 0, 8, -6, 2], [0, 0, -12, 8, -2],
              [0, 0, 8, -6, 2], [0, 0, -2, 2, -1]],
             dtype=np.float32),
    np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
]
filter_d = [
    np.array([[0, 0, 0, 0, 0], [0, -1, 2, 0, 0], [0, 2, -4, 0, 0],
              [0, -1, 2, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, 0, 2, -1, 0], [0, 0, -4, 2, 0],
              [0, 0, 2, -1, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
    np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
              [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
]


filter_e = [
    np.array([[ -1,   2,  -2,   0,   0],
       [  2,  -6,   8,   0,   0],
       [ -2,   8, -12,   0,   0],
       [  2,  -6,   8,   0,   0],
       [ -1,   2,  -2,   0,   0]],
             dtype=np.float32),
    np.array([[  0,   0,   0,   0,   0],
       [  0,   0,   0,   0,   0],
       [ -2,   8, -12,   8,  -2],
       [  2,  -6,   8,  -6,   2],
       [ -1,   2,  -2,   2,  -1]],dtype=np.float32),
    np.array([[  0,   0,  -2,   2,  -1],
       [  0,   0,   8,  -6,   2],
       [  0,   0, -12,   8,  -2],
       [  0,   0,   8,  -6,   2],
       [  0,   0,  -2,   2,  -1]],dtype=np.float32),
    np.array([[ -1,   2,  -2,   2,  -1],
       [  2,  -6,   8,  -6,   2],
       [ -2,   8, -12,   8,  -2],
       [  0,   0,   0,   0,   0],
       [  0,   0,   0,   0,   0]],dtype=np.float32),

]
filter_f = [
    np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
              [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
             dtype=np.float32),
]
filter_g = [
    np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
              [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
             dtype=np.float32),
]
def _apply_filters(src, filters, divisor):
    src_copy = np.copy(src)
    img = cv2.filter2D(src=src_copy, kernel=filters[0], ddepth=-1)
    for filter in filters[1:]:
        img = cv2.add(img, cv2.filter2D(src=src_copy, kernel=filter, ddepth=-1))
    return img //divisor
def apply_filters(rich_texture, poor_texture):
    rich_img = _apply_filters(rich_texture, filter_a, 8)+_apply_filters(rich_texture, filter_b, 8)+_apply_filters(rich_texture, filter_c, 4)+_apply_filters(rich_texture, filter_d, 4)+_apply_filters(rich_texture, filter_e, 4)+_apply_filters(rich_texture, filter_f, 1)+_apply_filters(rich_texture, filter_g, 1)
    poor_img = _apply_filters(poor_texture, filter_a, 8)+_apply_filters(poor_texture, filter_b, 8)+_apply_filters(poor_texture, filter_c, 4)+_apply_filters(poor_texture, filter_d, 4)+_apply_filters(poor_texture, filter_e, 4)+_apply_filters(poor_texture, filter_f, 1)+_apply_filters(poor_texture, filter_g, 1)
    rich_img = cv2.cvtColor(rich_img, cv2.COLOR_RGB2GRAY)//7
    poor_img = cv2.cvtColor(poor_img, cv2.COLOR_RGB2GRAY)//7
    img_rich_thresh = np.median(rich_img)+2
    img_poor_thresh = np.median(poor_img)+2
    return cv2.threshold(rich_img, img_rich_thresh, 255, cv2.THRESH_BINARY)[1], cv2.threshold(poor_img, img_poor_thresh, 255, cv2.THRESH_BINARY)[1]
