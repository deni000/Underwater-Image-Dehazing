import numpy as np
import cv2
import matplotlib.pyplot as plt

def dark_channel(image, window_size):
    """Calculate the dark channel prior of an image."""
    if len(image.shape) < 3:  # Ensure image has at least 3 dimensions (height, width, channels)
        raise ValueError("Input image must be a color image (3 channels).")
    min_channel = np.min(image, axis=2)
    return cv2.erode(min_channel, np.ones((window_size, window_size), dtype=np.uint8))

def atmospheric_light(image, dark_channel):
    """Estimate the atmospheric light of an image."""
    if len(image.shape) < 3:
        raise ValueError("Input image must be a color image (3 channels).")
    flat_image = image.reshape((-1, 3))
    flat_dark = dark_channel.flatten()
    index = np.argsort(flat_dark)[::-1][:int(flat_dark.size * 0.001)]  # Take top 0.1% brightest pixels
    return np.max(flat_image[index], axis=0)

def transmission(image, atmospheric_light, omega=0.95, window_size=15):
    """Estimate the transmission map of an image."""
    normalized_image = image.astype(np.float32) / atmospheric_light.astype(np.float32)
    dark = dark_channel(normalized_image, window_size)
    return 1 - omega * dark

def dehaze(image, atmospheric_light, t, t0=0.1):
    """Dehaze the image using the transmission map."""
    t = np.maximum(t, t0)
    result = np.empty_like(image, dtype=np.float32)
    for i in range(3):
        result[..., i] = ((image[..., i].astype(np.float32) - atmospheric_light[i]) / t) + atmospheric_light[i]
    return np.clip(result, 0, 255).astype(np.uint8)

def dehaze_image(image_path, window_size=15, omega=0.95, t0=0.1):
    """Dehaze an image."""
    image = cv2.imread(image_path)
    dark = dark_channel(image, window_size)
    atmospheric = atmospheric_light(image, dark)
    t = transmission(image, atmospheric, omega, window_size)
    dehazed = dehaze(image, atmospheric, t, t0)
    return dehazed

# Prompt user for input image file
image_path = input("Enter the path to the input image file: ")

# Perform dehazing
try:
    dehazed_image = dehaze_image(image_path)
    plt.imshow(cv2.cvtColor(dehazed_image, cv2.COLOR_BGR2RGB))
    plt.title("Dehazed Image")
    plt.show()
except Exception as e:
    print("Error:", e)

