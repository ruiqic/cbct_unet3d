import os
import logging
import socket


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def get_nifti_file_list(root):
    """
    Gets a sorted list of file paths ending in '.nii.gz'.
    Paths are full paths.
    """
    
    fns = sorted(filter(lambda x: x.endswith(".nii.gz"), os.listdir(root)))
    files = [os.path.join(root, x) for x in fns]
    return files

def find_free_port():
    """
    Find a free network port.

    Returns:
        int: A free network port number.
    """
    # Create a socket object
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to an address with a random port number
    sock.bind(('localhost', 0))

    # Get the port number assigned by the operating system
    _, port = sock.getsockname()

    # Close the socket
    sock.close()

    return port