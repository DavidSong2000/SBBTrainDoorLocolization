import socket
import struct
import signal
import sys
from PIL import Image
import io

from matplotlib import pyplot as plt
from ultralytics import YOLO

from yoloBox.scripts.single_img_bounding import single_img_bounding

# from limap.scripts import LoadReconstruction

# Global variable for reconstruction data
reconstruction_data = None

# Flag to indicate whether the server should continue running
running = True

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

def handle_client(client_socket):
    """
    Handle a single client connection.
    """
    try:
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
            print(f"Received packet of length: {len(packet)} bytes")
            if  not packet:
                break
            image_data += packet
            print(f"Total received: {len(image_data)} bytes")


        print(f"Received image data of length: {len(image_data)} bytes")

        # Decode the image
        image = Image.open(io.BytesIO(image_data))
        # save the image to script
        image.save('image.png')




        # Run the localization process
        result = process_localization(image)

        # Send response to the client
        response = str(result).encode('utf-8')
        client_socket.sendall(response)

    except Exception as e:
        print(f"Error handling connection: {e}")
        error_response = {"status": "error", "message": str(e)}
        client_socket.sendall(str(error_response).encode('utf-8'))

    finally:
        client_socket.close()

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
        single_img_bounding(bounded_pic, model)
        bounded_pic.save('bounded_pic.png')

        # Step 2: Run LIMAP Localization using the reconstruction data
        # Replace the following placeholder with your actual localization logic
        # TODO: LIMAP REAL TIME LOCALIZATION（feature matching）API
        localization_result = {"x": 100, "y": 200, "width": 50, "height": 50}

        return {"status": "success", "localization": localization_result}

    except Exception as e:
        print(f"Error during processing: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    # Register the signal handler to handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    start_server()
