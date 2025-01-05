import socket
import struct
import signal
import sys
from PIL import Image
import io
import os 
import time
import json
import threading

from matplotlib import pyplot as plt
from ultralytics import YOLO

from yoloBox.scripts.single_img_bounding import single_img_bounding

# Info for saving folder
time_now = time.strftime("%Y%m%d-%H_%M", time.localtime())
# main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
main_path = os.getcwd()
current_log_path = os.path.join(main_path, 'Logs/' + time_now)

# from limap.scripts import LoadReconstruction

# Global variable for reconstruction data
reconstruction_data = None

# Flag to indicate whether the server should continue running
running = True

# Index for client
client_idx = 0

def signal_handler(sig, frame):
    """
    Handle Ctrl+C signal to cleanly shutdown the server.
    """
    global running
    print("Shutting down server...")
    running = False

def load_reconstruction():
    """
    Load pre-trained reconstruction data.
    """
    print("Loading LIMAP reconstruction data...")
    # TODO: LIMAP PRE-TRAINED RECONSTRUCTION (Feature reconstruction) API
    # data = LoadReconstruction()  # Replace with actual function to load your data
    data = 0
    print("Reconstruction data loaded successfully.")
    return data

def start_server(host='0.0.0.0', port=5001):
    """
    Start a TCP/IP server to handle front-end requests.
    """
    global reconstruction_data, running

    # Load reconstruction data
    # reconstruction_data = load_reconstruction()

    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)  # Listen for up to 5 connections
    server_socket.settimeout(1.0)  # Set a timeout for the server socket
    print(f"Server started. Listening on {host}:{port}")

    while running:
        try:
            print("Waiting for a connection...")
            client_socket, client_address = server_socket.accept()
            print(f"Connection from {client_address}")

            handle_client(client_socket)
        except socket.timeout:
            # This allows the loop to continue and check the `running` flag
            continue
        except Exception as e:
            print(f"Error in server loop: {e}")

    print("Server has been stopped.")
    server_socket.close()
    
def receive_data(client_socket, length):
    data = bytearray()
    while len(data) < length:
        packet = client_socket.recv(length - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def handle_client(client_socket):
    """
    Handle a single client connection.
    """
    global client_idx
    try:
        camera_info_recieved = {}
        # receive camera intrinsics 4*4
        intrinsics_data = receive_data(client_socket, 4 * 4)
        camera_focal_length = struct.unpack('ff', intrinsics_data[:8])
        camera_principal_point = struct.unpack('ff', intrinsics_data[8:])
        if camera_focal_length is None or camera_principal_point is None:
            print("Failed to receive camera intrinsics")
            client_socket.close()
            return
        print(f'Camera Focal Length: {camera_focal_length}')
        print(f'Camera Principal Point: {camera_principal_point}')
        # Save to camera info recieved dict
        camera_info_recieved['camera_focal_length'] = camera_focal_length
        camera_info_recieved['camera_principal_point'] = camera_principal_point
        
        # # Receive camera pose data (12 bytes for position, 16 bytes for rotation)
        pose_data = receive_data(client_socket, 7 * 4)  # 7 floats, each float is 4 bytes
        camera_position = struct.unpack('fff', pose_data[:12])
        camera_rotation = struct.unpack('ffff', pose_data[12:])
        if camera_position is None or camera_rotation is None:
            print("Failed to receive camera pose")
            client_socket.close()
            return
        print(f'Camera Position: {camera_position}')
        print(f'Camera Rotation: {camera_rotation}')
        # Save to camera info recieved dict
        camera_info_recieved['camera_position'] = camera_position
        camera_info_recieved['camera_rotation'] = camera_rotation
        
        # Receive image length (4 bytes)
        length_data = client_socket.recv(4)
        if not length_data:
            print("No length data received. Closing connection.")
            client_socket.close()
            return

        # Unpack the image length
        image_length = struct.unpack('!I', length_data)[0]
        print(f"Expecting image data of length: {image_length} bytes")

        # Receive the image data
        image_data = b""
        while len(image_data) < image_length:
            packet = client_socket.recv(4096) # Receive the remaining data
            # print(f"Received packet of length: {len(packet)} bytes")
            if not packet:
                break
            image_data += packet
            # print(f"Total received: {len(image_data)} bytes / {image_length} bytes")

        print(f"Received image data of length: {len(image_data)} bytes")

        # Decode the image
        image = Image.open(io.BytesIO(image_data))
        # save the image to Log
        image_name = f'Client{client_idx}_rawImage.png'
        image.save(os.path.join(current_log_path, image_name))
        # save current image
        current_image = image
        current_image_name = 'rawimage00001000.png'
        current_image.save(os.path.join(os.getcwd(), current_image_name))

        # save camera information to json
        recieve_json_name = f'Client{client_idx}_cameraInfoRecieved.json'
        with open(os.path.join(current_log_path, recieve_json_name), 'w') as f:
            json.dump(camera_info_recieved, f, sort_keys=False, indent=4)

        # Run the localization process
        result = None
        def localization_thread(image):
            nonlocal result
            result = communicate_with_remote_server(image)
        
        localization_thread_instance = threading.Thread(target=localization_thread, args=(image,))
        localization_thread_instance.start()
        localization_thread_instance.join()  # Wait for the result

        if result:
            # Send response to the client
            response = json.dumps(result).encode('utf-8')
            client_socket.sendall(response)

            client_idx += 1
        else:
            print("Can't fetch localization result")


    except Exception as e:
        print(f"Error handling connection: {e}")
        error_response = {"status": "error", "message": str(e)}
        client_socket.sendall(str(error_response).encode('utf-8'))

    finally:
        client_socket.close()

def communicate_with_remote_server(image):
    """
    Communicate with the remote server to process the image.
    """
    try:
        # Convert image to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()

        # Connect to the remote localization server
        remote_host = '192.168.1.123'  # Replace with the remote server's IP
        remote_port = 6001  # Replace with the remote server's port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as remote_socket:
            remote_socket.connect((remote_host, remote_port))

            # Send image length and image data
            remote_socket.sendall(struct.pack('!I', len(image_bytes)))
            remote_socket.sendall(image_bytes)

            # Receive response data
            response_data = b""
            packet = remote_socket.recv(48)
            response_data += packet

        # Decode and return the response
        localization_result = json.loads(response_data.decode('utf-8'))
        return localization_result

    except Exception as e:
        print(f"Error during remote localization: {e}")
        return {"status": "error", "message": str(e)}
    

def process_localization(image):
    """
    Process the image using YOLO and LIMAP for localization.
    """
    try:
        # Step 1: Run YOLO to filter the image
        # bounded_pic = PicBounding(image)
        # TODO: YOLO API
        model = YOLO('yoloBox/weights/best.pt')
        bounded_pic = image
        bounded_pic= single_img_bounding(bounded_pic, model)
        bounded_pic_name = f'Client{client_idx}_boundedImage.png'
        bounded_pic.save(os.path.join(current_log_path, bounded_pic_name))
        current_bounded_pic = bounded_pic
        current_bounded_pic_name = 'image00001000.png'
        current_bounded_pic.save(os.path.join(os.getcwd(), current_bounded_pic_name))


        # Step 2: Run LIMAP Localization using the reconstruction data
        # Replace the following placeholder with your actual localization logic
        # TODO: LIMAP REAL TIME LOCALIZATION（feature matching）API
        localization_result = [[0.0, 0.0, 0.0], 
                               [0.0, 0.0, 0.0], 
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]

        # save result information to json
        result_json_name = f'Client{client_idx}_localizationResult.json'
        with open(os.path.join(current_log_path, result_json_name), 'w') as f:
            json.dump(localization_result, f, sort_keys=False, indent=4)

        return localization_result
        # return {"status": "success", "localization": localization_result}

    except Exception as e:
        print(f"Error during processing: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    # Create Saving Folder
    if os.path.exists(current_log_path):  # 看文件夹是否存在
        print('文件夹已存在')
    else:  # 如果不存在
        os.makedirs(current_log_path)  # 则创建文件夹
        print(f'Created Folder {current_log_path}')

    # Register the signal handler to handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    start_server()
