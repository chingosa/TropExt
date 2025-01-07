import os

def calculate_directory_size(directory):
    """
    Calculate the total size of all files in a directory and its subdirectories.
    
    Args:
        directory (str): Path to the directory.
        
    Returns:
        int: Total size in bytes.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"The provided path '{directory}' is not a valid directory.")
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Check if it's a file and add its size
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def human_readable_size(size_in_bytes):
    """
    Convert a size in bytes to a human-readable format (KB, MB, GB, etc.).
    
    Args:
        size_in_bytes (int): Size in bytes.
        
    Returns:
        str: Human-readable size.
    """
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} PB"

# Example usage

directory_path = os.getcwd()  # Replace with your directory path
total_size = calculate_directory_size(directory_path)
print(f"Total size: {human_readable_size(total_size)}")
