import socket
import pickle
import os
import cv2
import time
from ocr import tools
import numpy as np
import easyocr
# Server configuration
HOST = '130.194.71.74'  # Server's IP address
PORT = 65432        # Port used by the server

reader = easyocr.Reader(['en'], gpu=False) # this needs to run only once to load the model into memory

def ocr_preprocessing(image):
    # Slight sharpening using a kernel on each color channel
    kernel_sharpening = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])

    sharpened = cv2.merge([cv2.filter2D(image[:, :, i], -1, kernel_sharpening) for i in range(image.shape[2])])
    return sharpened

def sort_results_by_bounding_box(results):
    # Sort results by the top-left corner's Y-coordinate (index 1) and then X-coordinate (index 0) of the bounding box
    results.sort(key=lambda res: (min(res[0], key=lambda point: point[1])[1] + min(res[0], key=lambda point: point[0])[0]/15))
    return results

# while True:
    # try:
# Create a TCP/IP socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    
    server_socket.bind((HOST, PORT)) 

    # # If socket was just used then need to reallocate socket    
    # except OSError as e:
    #     if e.errno == 98:
    #         server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #         server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #         server_socket.bind((HOST, PORT))
    server_socket.listen()
    print("Server is waiting for a connection...")
    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")
    while True:
        # Receive folder name
        folder_name_length_bytes = conn.recv(2)
        folder_name_length = int.from_bytes(folder_name_length_bytes, byteorder='big')
        folder_name = conn.recv(folder_name_length).decode()

        # # Receive image name length and image name as bytes
        # image_name_length_bytes = server_socket.recv(2)
        # image_name_length = int.from_bytes(image_name_length_bytes, byteorder='big')
        # image_name = server_socket.recv(image_name_length).decode()

        # Receive file size
        file_size_bytes = conn.recv(4)
        file_size = int.from_bytes(file_size_bytes, byteorder='big')

        # Receive image data
        received_data = b''
        while len(received_data) < file_size:
            chunk = conn.recv(min(file_size - len(received_data), 4096))
            if not chunk:
                break
            received_data += chunk

        # Deserialize the received image data
        received_image = pickle.loads(received_data)
        os.makedirs("images", exist_ok=True)
        image_name = os.path.join("images", folder_name + ".png")

        # Save the received image with the given folder and image name
        received_image = cv2.rotate(received_image, cv2.ROTATE_180)
        received_image = ocr_preprocessing(received_image)
        cv2.imwrite(image_name, received_image)
        result = sort_results_by_bounding_box(tools.read_text(received_image, reader))
        for a,b,c in result:
            print(b, end=" ")

        print(f"Received and saved {os.path.join(folder_name, image_name)}")
    # except EOFError:
    #     server_socket.close()
    #     time.sleep(1)
    #     continue

server_socket.close()