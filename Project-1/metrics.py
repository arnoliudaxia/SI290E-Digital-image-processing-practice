import numpy as np
import cv2
from skimage import filters
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from scipy.ndimage import convolve
from skimage.measure import shannon_entropy

def metrics(Y1, Y2, which_metrics):

    # Read the images
    ipic = cv2.imread(Y1, cv2.IMREAD_GRAYSCALE)
    Y1 = np.float64(ipic)

    epic = cv2.imread(Y2, cv2.IMREAD_GRAYSCALE)
    Y2 = np.float64(epic)

    if which_metrics == "MAE":
        # MAE calculation
        MAE = np.mean(np.abs(Y1 - Y2))
        print(f"MAE: {MAE:2f}")
        return MAE  # 返回MAE值

    elif which_metrics == "GMSD":
        # GMSD calculation
        T = 170
        Down_step = 2
        dx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3
        dy = dx.T

        aveKernel = np.ones((2, 2)) / 4

        # Apply averaging (down-sampling) to each channel
        aveY1 = convolve(Y1, aveKernel)
        aveY2 = convolve(Y2, aveKernel)

        Y1_down = aveY1[::Down_step, ::Down_step]
        Y2_down = aveY2[::Down_step, ::Down_step]

        # Compute gradients
        gradientMap1 = np.sqrt(convolve(Y1_down, dx) ** 2 + convolve(Y1_down, dy) ** 2)
        gradientMap2 = np.sqrt(convolve(Y2_down, dx) ** 2 + convolve(Y2_down, dy) ** 2)

        # Compute quality map
        quality_map = (2 * gradientMap1 * gradientMap2 + T) / (gradientMap1 ** 2 + gradientMap2 ** 2 + T)

        # GMSD score (standard deviation of the quality map)
        GMSD = np.std(quality_map)
        print(f"GMSD: {GMSD}")
        return GMSD  # 返回GMSD值
    
    elif which_metrics == "NIQE":
        # NIQE Calculation (Assuming NIQE function is available or using a pre-defined function)
        # You can use a pre-implemented NIQE calculation function or a library like `pyimagequality`
        print("Use matlab code to calculate NIQE!")
        return None


# Call the metrics function
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python metrics.py <path_to_Y1> <path_to_Y2>")
        sys.exit(1)
    Y1_path = sys.argv[1]
    Y2_path = sys.argv[2]
    metrics(Y1_path, Y2_path)
