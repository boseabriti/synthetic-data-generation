from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# load image from the IAM database
# url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")


# image_path = '/Users/abriti.bose/Documents/Abriti-openAI/testimage.png'
# image = Image.open(image_path).convert('RGB')

# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')



# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')

# processor.save_pretrained('test')
# model.save_pretrained('test')

# pixel_values = processor(images=image, return_tensors="pt").pixel_values

# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_text)



# load image from the IAM database (actually this model is meant to be used on printed text)
image_path = '/Users/abriti.bose/Documents/Abriti-openAI/comedy.png'
image = Image.open(image_path).convert('RGB')

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
processor.save_pretrained('/Users/abriti.bose/Documents/Abriti-openAI/test')
model.save_pretrained('/Users/abriti.bose/Documents/Abriti-openAI/test')
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)