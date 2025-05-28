import numpy as np
from skimage import io,color

#Calculate the dice coefficient of two binary images

def dice_coefficient(image1, image2):
    intersection = np.logical_and(image1, image2)
    return 2. * intersection.sum() / (image1.sum() + image2.sum())

#example use
gt_path = r"project5\images\fingerprint_mask.jpg"
gt = io.imread(gt_path, as_gray=True) > 0.5

image = io.imread(r"project5\images\fingerprint.jpg")
gray_image = color.rgb2gray(image)
feature_map = np.zeros_like(gray_image,dtype=np.float32) 
mask = (feature_map >= 0) #threshold value

dice = dice_coefficient(gt, mask)
print(f"Dice Coefficient: {dice:.4f}")