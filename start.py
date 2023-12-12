import socket
import pickle
import os
import cv2

# Server configuration
HOST = '130.194.71.78'  # Server's IP address
PORT = 65432        # Port used by the server


# Create a TCP/IP socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((HOST, PORT))

    while True:

        # Receive folder name
        folder_name_length_bytes = client_socket.recv(2)
        folder_name_length = int.from_bytes(folder_name_length_bytes, byteorder='big')
        folder_name = client_socket.recv(folder_name_length).decode()

        # Receive image name length and image name as bytes
        image_name_length_bytes = client_socket.recv(2)
        image_name_length = int.from_bytes(image_name_length_bytes, byteorder='big')
        image_name = client_socket.recv(image_name_length).decode()

        # Receive file size
        file_size_bytes = client_socket.recv(4)
        file_size = int.from_bytes(file_size_bytes, byteorder='big')

        # Receive image data
        received_data = b''
        while len(received_data) < file_size:
            chunk = client_socket.recv(min(file_size - len(received_data), 4096))
            if not chunk:
                break
            received_data += chunk

        # Deserialize the received image data
        received_image = pickle.loads(received_data)

        folder_path = os.path.join("images", folder_name)
        
        # Create folders if they don't exist
        os.makedirs(folder_path, exist_ok=True)

        # Save the received image with the given folder and image name
        image_path = os.path.join(folder_path, image_name)
        cv2.imwrite(image_path + ".png", received_image)

        print(f"Received and saved {os.path.join(folder_name, image_name)}")