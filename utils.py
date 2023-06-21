import cv2
import numpy as np


def triplane2img(tri_path, out_path):
    triplanes = np.load(tri_path).reshape(3, 16, 128, 128)
    tri_img = triplanes.reshape(3, 512, 512).transpose(2,1,0)
    tri_img -= np.min(tri_img)
    tri_img *= 255.0/np.max(tri_img)
    cv2.imwrite(out_path, tri_img)
