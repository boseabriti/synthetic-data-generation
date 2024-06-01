from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from tqdm.auto import tqdm
from urllib.request import urlretrieve
from zipfile import ZipFile
import requests
 
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import glob

#test pytorch installation
# import torch
# x = torch.rand(5, 3)
# print(x)

device = torch.device("cpu")
# device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')


#function to download and extract the dataset from IAM
# def download_and_unzip(url, save_path):
#     print(f"Downloading and extracting assets....", end="")
 
 
#     # Downloading zip file using urllib package.
#     urlretrieve(url, save_path)
 
 
#     try:
#         # Extracting zip file using the zipfile package.
#         with ZipFile(save_path) as z:
#             # Extract ZIP file contents in the same directory.
#             z.extractall(os.path.split(save_path)[0])
 
 
#         print("Done")
 
 
#     except Exception as e:
#         print("\nInvalid file.", e)
 
# URL = r"/Users/abriti.bose/Documents/Abriti-openAI/testimage.png"
# asset_zip_path = os.path.join(os.getcwd(), "images.zip")
# # Download if assest ZIP does not exists.
# if not os.path.exists(asset_zip_path):
#     download_and_unzip(URL, asset_zip_path)


#function to read an image in PIL format and return it for the next processing stage.
def read_image():
    """
    :param image_path: String, path to the input image.
 
 
    Returns:
        image: PIL Image.
    """
    # image_path = '/Users/abriti.bose/Documents/Abriti-openAI/testimage.png'
    # image = Image.open(image_path).convert('RGB')

    url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image



#helper function to carry out the OCR pipeline.
def ocr(image, processor, model):
    """
    :param image: PIL Image.
    :param processor: Huggingface OCR processor.
    :param model: Huggingface OCR model.
 
 
    Returns:
        generated_text: the OCR'd text string.
    """
    # We can directly perform OCR on cropped images.
    pixel_values = processor(image, return_tensors='pt').pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


#helper function that runs inference on new images. It combines the previous functions and shows the images in the output cell.
def eval_new_data(data_path=None, num_samples=4, model=None):
    image_paths = glob.glob(data_path)
    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        if i == num_samples:
            break
        image = read_image(image_path)
        text = ocr(image, processor, model)
        plt.figure(figsize=(7, 4))
        plt.imshow(image)
        plt.title(text)
        plt.axis('off')
        plt.show()


# #Inference on Printed Text
# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
# model = VisionEncoderDecoderModel.from_pretrained(
#     'microsoft/trocr-small-printed'
# ).to(device)



# eval_new_data(
#     data_path=os.path.join('images', 'newspaper', '*'),
#     num_samples=2,
#     model=model
# )


#Inference on Handwritten Text
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained(
    'microsoft/trocr-base-handwritten'
).to(device)
# pixel_values = processor(images=image, return_tensors="pt").pixel_values

# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_text)

eval_new_data(
    data_path=os.path.join('images', 'handwritten', '*'),
    num_samples=2,
    model=model
)