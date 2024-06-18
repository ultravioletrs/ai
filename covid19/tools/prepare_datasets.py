import argparse
import zipfile
import os
import shutil
import random
import tempfile

def unzip_file(zip_filename, extract_to):
    """
    Unzips the given zip file into the specified directory.
    
    Parameters:
    zip_filename (str): The path to the zip file.
    extract_to (str): The directory to extract the files to.
    
    Returns:
    str: Path to the directory where files are extracted.
    """
    if not zipfile.is_zipfile(zip_filename):
        print(f"Error: '{zip_filename}' is not a valid zip file.")
        return None
    
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Unzipped '{zip_filename}' to '{extract_to}'")
        return extract_to
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def create_structure(base_dir):
    """
    Creates the directory structure with subdirectories: h1, h2, h3, and test.
    
    Parameters:
    base_dir (str): The base directory where the structure should be created.
    """
    subdirs = ['h1', 'h2', 'h3', 'test']
    categories = ['COVID', 'Normal', 'Viral Pneumonia']
    
    for subdir in subdirs:
        for category in categories:
            os.makedirs(os.path.join(base_dir, subdir, category), exist_ok=True)

def copy_images(source_dir, target_dir):
    """
    Copies images from source_dir to target_dir maintaining the structure.
    
    Parameters:
    source_dir (str): The directory containing the images in subdirectories.
    target_dir (str): The base directory where the new structure is created.
    """
    top_level_dir = os.path.join(source_dir, 'COVID-19_Radiography_Dataset')
    categories = ['COVID', 'Normal', 'Viral Pneumonia']
    subdirs = ['h1', 'h2', 'h3', 'test']
    
    for category in categories:
        src_images_dir = os.path.join(top_level_dir, category, 'images')
        if not os.path.isdir(src_images_dir):
            print(f"Error: '{src_images_dir}' does not exist.")
            continue
        
        images = os.listdir(src_images_dir)
        random.shuffle(images)
        num_images = len(images)
        num_per_subdir = num_images // len(subdirs)
        extra = num_images % len(subdirs)
        
        idx = 0
        for subdir in subdirs:
            start_idx = idx * num_per_subdir + min(idx, extra)
            end_idx = start_idx + num_per_subdir + (1 if idx < extra else 0)
            for image in images[start_idx:end_idx]:
                src_image_path = os.path.join(src_images_dir, image)
                dest_image_path = os.path.join(target_dir, subdir, category, image)
                shutil.copy(src_image_path, dest_image_path)
            idx += 1

def main():
    parser = argparse.ArgumentParser(description='Unzip a zip file and organize images.')
    parser.add_argument('zipfile', type=str, help='The path to the zip file to unzip')
    parser.add_argument('-d', '--destination', type=str, default='data', help='The directory to extract the files to (default is data)')

    args = parser.parse_args()

    zip_filename = args.zipfile
    destination = args.destination

    with tempfile.TemporaryDirectory() as tmpdirname:
        if unzip_file(zip_filename, tmpdirname):
            data_dir = os.path.join(os.getcwd(), destination)
            create_structure(data_dir)
            copy_images(tmpdirname, data_dir)

    print(f"Data directory '{destination}' is ready and contains the datasets.")

if __name__ == "__main__":
    main()
