import os
from sklearn.model_selection import train_test_split
from PIL import Image

OPENSLIDE_PATH = r'C:\Users\Acer\miniconda3\envs\UniProject\Lib\site-packages\openslide\openslide-win64-20231011\bin'

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

def read_slide_patches(slide_path, patch_size=(256, 256), overlap=0.1):
    try:
        slide = openslide.OpenSlide(slide_path)
        width, height = slide.dimensions

        # Define patch coordinates with overlap
        patch_width = int(patch_size[0] * (1 - overlap))
        patch_height = int(patch_size[1] * (1 - overlap))

        # Extract patches without reading the entire region into memory
        patches = []
        for x in range(0, width - patch_width, patch_width):
            for y in range(0, height - patch_height, patch_height):
                region = slide.read_region((x, y), 0, patch_size)
                patch = Image.new("RGB", patch_size)
                patch.paste(region, (0, 0))
                patches.append(patch)

        print(f"Extracted {len(patches)} patches from slide: {slide_path}")
        return patches
    except openslide.OpenSlideUnsupportedFormatError as e:
        print(f"Error opening slide {slide_path}: {e}")
        return None


def split_data(data_folder, test_size=0.2, random_state=42):
    image_files = []
    annotation_files = []

    for subfolder in os.listdir(data_folder):
        subfolder_path = os.path.join(data_folder, subfolder)

        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)

                if filename.endswith((".svs", ".mrxs", ".ndpi", ".tif")):
                    image_files.append(file_path)
                    annotation_files.append(None)  # Placeholder for missing annotations

    # Split the data into training and validation sets
    image_train, image_valid, annotation_train, annotation_valid = train_test_split(
        image_files, annotation_files, test_size=test_size, random_state=random_state
    )

    return image_train, image_valid, annotation_train, annotation_valid
