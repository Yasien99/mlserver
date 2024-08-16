from torchvision import models, transforms
from inspect import iscoroutinefunction
from functools import wraps, partial
from pydantic import BaseModel
from fastapi import FastAPI
from torch import optim
from io import BytesIO
import torch.nn as nn
from PIL import Image
import numpy as np
import requests
import logging 
import uvicorn
import easyocr
import torch
import time
import cv2
import re 
app = FastAPI()


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_execution_time(func=None, *, is_async=False):
    if func is None:
        return partial(log_execution_time, is_async=is_async)

    if iscoroutinefunction(func) or is_async:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Started {func.__name__} with args: {args} and kwargs: {kwargs}")
            result = await func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.debug(f"Finished {func.__name__} in {execution_time:.4f} seconds")
            return result
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Started {func.__name__} with args: {args} and kwargs: {kwargs}")
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.debug(f"Finished {func.__name__} in {execution_time:.4f} seconds")
            return result
        return sync_wrapper

class ImageURLs(BaseModel):
    urls: list[str]


class EfficientNetBinary(nn.Module):
    def __init__(self):
        super(EfficientNetBinary, self).__init__()
        self.base_model = models.efficientnet_b2(pretrained=True)
        self.base_model.classifier = nn.Sequential(
            nn.Linear(self.base_model.classifier[1].in_features, 4),
            nn.Softmax(dim=1)  # Use Softmax for multi-class classification
        )

    def forward(self, x):
        return self.base_model(x)

# Initialize the model
model = EfficientNetBinary()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)  # Weight decay for L2 regularization
checkpoint = torch.load('best_model_EffcientNet_4C.pth',  map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epochs']
model.class_to_idx = checkpoint['class_to_idx']
model.idx_to_class = checkpoint['idx_to_class']
model.eval()

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Define the image transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

@log_execution_time
def preprocess_image(image: Image.Image):
    image = transform(image)
    image = image.unsqueeze(0)
    return image

@log_execution_time
def perform_ocr_on_claim(image: Image.Image):
    # Convert PIL image to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize the image to 600x600 pixels
    resized_image = cv2.resize(image, (600, 600))

    # Select the ROI manually (adjust coordinates as needed)
    x, y, w, h = 0, 0, 550, 80
    roi = resized_image[y:y+h, x:x+w]

    # Perform OCR on the cropped image
    results = reader.readtext(roi)

    # Extract the numbers matching the regular expression
    number_pattern = re.compile(r'^\d+$')
    extracted_numbers = [text for (bbox, text, prob) in results if number_pattern.match(text)]

    return extracted_numbers

@log_execution_time
def perform_ocr_on_crop_approval_number(image: Image.Image):
    # Convert PIL image to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize the image to 1000x1000 pixels
    resized_image = cv2.resize(image, (1000, 1000))

    # Select the ROI manually (adjust coordinates as needed)
    x, y, w, h = 0, 100, 1000, 250
    roi = resized_image[y:y+h, x:x+w]

    # Perform OCR on the cropped image
    results = reader.readtext(roi)

    # Extract the numbers matching the regular expression
    number_pattern = re.compile(r'^\d{6,}$')
    extracted_numbers = [text for (bbox, text, prob) in results if number_pattern.match(text)]

    return extracted_numbers

@log_execution_time
def perform_ocr_on_crop_approval_card_number(image: Image.Image):
    # Convert PIL image to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Resize the image to 600x600 pixels
    resized_image = cv2.resize(image, (1000, 1000))

    # Select the ROI manually (adjust coordinates as needed)
    x, y, w, h = 500, 280, 1000, 150
    roi = resized_image[y:y+h, x:x+w]

    # Perform OCR on the cropped image
    results = reader.readtext(roi)

    # Extract the numbers matching the regular expression
    number_pattern = re.compile(r'^\d{6,}$')
    extracted_numbers = [text for (bbox, text, prob) in results if number_pattern.match(text)]

    return extracted_numbers

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
@log_execution_time
async def predict(images: ImageURLs):
    results = []
    for url in images.urls:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image = preprocess_image(image)
            image = image.to(device)
            
            with torch.no_grad():
                output = model(image)

                topk, topclass = output.topk(3, dim=1)

                # Extract the actual classes and probabilities
                top_classes = [
                    model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
                ]
                image_class = top_classes[0]

                top_p = topk.cpu().numpy()[0]
                probability = str(top_p[0])

                # Print the top class probability and its label
                if image_class == "Claims":
                    original_image = Image.open(BytesIO(response.content)).convert("RGB")
                    extracted_numbers = perform_ocr_on_claim(original_image)
                    prescription_code = ", ".join(extracted_numbers)
                    results.append({
                        "url": url,
                        "class": image_class,
                        "probability": probability,
                        "prescription_code ": prescription_code
                    })                    

                elif image_class == "Approval":
                    original_image = Image.open(BytesIO(response.content)).convert("RGB")
                    extracted_numbers = perform_ocr_on_crop_approval_card_number(original_image)
                    card_number = extracted_numbers[0] if extracted_numbers else ""

                    extracted_numbers = perform_ocr_on_crop_approval_number(original_image)
                    approval_number = extracted_numbers[0] if extracted_numbers else ""

                    results.append({
                        "url": url,
                        "class": image_class,
                        "probability": probability,
                        "approval_number": approval_number,
                        "card_number": card_number
                    })
                elif image_class == "Receipt":
                    results.append({
                        "url": url,
                        "class": image_class,
                        "probability": probability
                    })
                elif image_class == "Random":
                    results.append({
                        "url": url,
                        "class": image_class,
                        "probability": probability
                    })
        else:
            results.append({
                "url": url,
                "error": f"Failed to download image. HTTP status code: {response.status_code}"
            })

    return results

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
