from PIL import Image
import os
from tqdm import tqdm
import glob
import shutil
import numpy as np

# Basic Camera Properties
CAPTURE_WIDTH = 640 # Resolution to capture W
CAPUTRE_HEIGHT = 480 # Resolution to capture H
CAPUTRE_RATE = 0.3 # Sample images every CAPTURE_RATE seconds. If there is an ANOMALY, capture at ANOMALY_CAPTURE_RATE

# Anomaly Related Parameters
ANOMALY_CAPTURE_RATE = 1 # Sample images every x seconds during an anomaly. These images will be storeqd
MINIMUM_ANOMALY_DURATION = 5 # Minimum anomaly duration in frames
MAXIMUM_ANOMALY_DURATION = 30
IMAGE_DIFFERENCE_BUFFER_LEN = 100 # Size of Buffer used to determine the if an image is an anomaly
ANOMALY_DIF_BUFFER_LEN = 10 # Size of Buffer used to determine if anomaly is no more significant
ANOMALY_SIGNIF = 5 # Frame triggers anomaly if frame_dif > ANOMALY_SIGIF * average frame_dif (image dif buffer) in buffer
ANOMALY_SIGNIF_END = 3 # Anomaly ends if frame_dif < ANOMALY_SIGIF_END * average frame_dif (anomaly dif buffer) in buffer
MINIMUM_TIME_BETWEEN_ANOMALY = 30 # An anomaly cannot be triggered to close to a previous one. (in frames)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            try:
                img = Image.open(filepath)
                images.append(img)
            except:
                print(f"Could not load {filepath}")
    return images

def calculate_sum_of_squared_differences(images):
 
    num_images = len(images)
    sum_sq_diff = 0

    # Convert all images to grayscale and ensure same size
    gray_images = [img.convert('L') for img in images]

    # Convert images to numpy arrays
    img_arrays = [np.array(img) for img in gray_images]

    for i in range(num_images - 1):
        for j in range(i + 1, num_images):
            # Calculate sum of squared differences using NumPy array operations
            sq_diff = np.sum((img_arrays[i] - img_arrays[j]) ** 2)
            sum_sq_diff += sq_diff

    return sum_sq_diff / num_images


def compute_average_and_return_sig(folder_path):
    significant_folders = []
    all_folders  = []
    
    print("Loading Images and calculating difference ")
    for folder in tqdm(glob.glob(folder_path)):
        images = load_images_from_folder(folder)
        if len(images) == 0:
            continue
        ssum_diff = calculate_sum_of_squared_differences(images)
        all_folders.append((folder, ssum_diff))
    
    average = sum(value for folder, value in all_folders) / len(all_folders)
    for folder, value in all_folders:
        if value > average:
            significant_folders.append(folder)
    
    return significant_folders

def process_images_and_copy(source_directory, destination_directory):
    
    source_directory_glob = source_directory + "*"
    folders_to_copy = compute_average_and_return_sig(source_directory_glob)
    
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    print("Copying Signifcant folders.")
    for folder_path in folders_to_copy:
        # Extract folder name from the full path
        folder_name = os.path.basename(folder_path)
        
        source_folder = os.path.join(source_directory, folder_path)
        destination_folder = os.path.join(destination_directory, folder_name)
        
        # Check if the folder exists in the source directory
        if os.path.exists(source_folder):
            try:
                # Copy the folder and its contents recursively
                shutil.copytree(source_folder, destination_folder)
                print(f"Folder '{folder_name}' copied successfully.")
            except shutil.Error as e:
                print(f"Error copying folder '{folder_name}': {e}")
        else:
            print(f"Folder '{folder_name}' does not exist in the source directory.")

# Example usage:
folders_path = 'C:/Users/abhin/Desktop/rpi_images/images3/images/'  # Path to the folder containing folders with images
destination = 'C:/Users/abhin/Desktop/rpi_images/images_filtered/'
process_images_and_copy(folders_path, destination)