from tenacity import retry, wait_random_exponential, stop_after_attempt
import requests
import os,sys,yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../src"))
config_file_path = os.path.join(project_root, "config", "config.yaml")
from trusteval import ModelService

with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

def call_gpt4o_api(prompt):
    service = ModelService(
        request_type="llm",
        handler_type="api",
        model_name="gpt-4o",
        config_path=config_file_path,
        temperature=0.7,
        max_tokens=2048
    )

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = service.process(prompt=prompt)
            # Check if the response is None or problematic
            if response is not None:
                return response

        except Exception as e:
            # Print error message and continue trying
            print(f"Attempt {attempt + 1}/{max_attempts} failed with error: {e}")
            if attempt == max_attempts - 1:
                # If maximum attempts reached, return an empty string
                return ""

    return ""

def generate_image(prompt):
    """
    Helper function to generate a single image and save it to an output path.
    Includes error handling to prevent failure from stopping the whole process.

    Args:
        prompt (str): The prompt string for image generation.
        output_path (str): The absolute file path to save the generated image.
        service (ModelService): An instance of ModelService to handle the generation.
    """
    service = ModelService(
        request_type='t2i',
        handler_type='api',
        model_name="dalle3",
        config_path=config_file_path,
    )

    try:
        result = service.process(prompt)
        if result is None:
            raise ValueError(f"No image generated for prompt: {prompt}")
        return result
    except Exception as e:
        print(f"Error generating image for prompt: '{prompt}' - {e}")

def generate_and_save_image(prompt, img_id, save_path):
    try:
        # Generate image
        image = generate_image(prompt=prompt)

        # Ensure the save directory exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Define the image file name
        file_name = f"{img_id}.png"
        full_path = os.path.join(save_path, file_name)

        # Save the image to the specified path
        image.save(full_path, format='PNG')

        #print(f"Image successfully saved to: {full_path}")
        return True

    except Exception as e:
        print(f"Image save failed: {str(e)}")
        return False
    
import os
from PIL import Image

def compress_image(input_path, output_path, quality=85, max_size_kb=100, max_attempts=20):
    """Compress a single image to JPG format with size limit"""
    try:
        with Image.open(input_path) as img:
            # Convert RGBA, P or other modes to RGB
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            output_path = os.path.splitext(output_path)[0] + '.jpg'

            for attempt in range(max_attempts):
                img.save(output_path, "JPEG", quality=quality)

                if os.path.getsize(output_path) <= max_size_kb * 1024:
                    print(f"Image compressed successfully to JPG in {attempt + 1} attempts: {output_path}")
                    break

                quality -= 5
                if quality < 20:
                    print(f"Cannot compress {input_path} to desired size. Saving with minimum quality.")
                    img.save(output_path, "JPEG", quality=20)
                    break
            else:
                print(f"Image could not be compressed to {max_size_kb}KB after {max_attempts} attempts: {output_path}")

    except IOError:
        print(f"Cannot open image file: {input_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def compress_images_in_folder(folder_path, output_folder, quality=85, max_size_kb=100, max_workers=4):
    """Compress all images in a folder using multithreading and save them in the specified output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images_to_compress = [
        (os.path.join(folder_path, filename), os.path.join(output_folder, filename))
        for filename in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, filename)) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(compress_image, input_path, output_path, quality, max_size_kb)
            for input_path, output_path in images_to_compress
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"An image compression task generated an exception: {exc}")

def merge_images(image_paths, output_path):
    """Merge multiple images horizontally into one image"""
    images = [Image.open(path).convert('RGB') for path in image_paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(output_path)