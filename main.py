from torchvision import models, transforms
from pydantic import BaseModel
from fastapi import FastAPI
from io import BytesIO
import torch.nn as nn
from PIL import Image
import numpy as np
import requests
import uvicorn
import easyocr
import torch
import cv2
import re 
app = FastAPI()

class ImageURLs(BaseModel):
    urls: list[str]


class EfficientNetBinary(nn.Module):
    def __init__(self):
        super(EfficientNetBinary, self).__init__()
        self.base_model = models.efficientnet_b2(pretrained=True)
        self.base_model.classifier = nn.Sequential(
            nn.Linear(self.base_model.classifier[1].in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

# Initialize the model
model = EfficientNetBinary()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load('efficientnet_model.pth', map_location=device))
model.eval()

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Define the image transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def preprocess_image(image: Image.Image):
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def perform_ocr_on_cropped_image(image: Image.Image):
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

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
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
                probability = output.item()
                image_class = "Approval" if probability > 0.5 else "Claim"
                
                if image_class == "Claim":
                    # Perform OCR if the image is classified as "Approval"
                    original_image = Image.open(BytesIO(response.content)).convert("RGB")
                    extracted_numbers = perform_ocr_on_cropped_image(original_image)
                    ocr_result = ", ".join(extracted_numbers)
                else:
                    ocr_result = "N/A"

            results.append({
                "url": url,
                "class": image_class,
                "probability": probability,
                "ocr_result": ocr_result
            })
        else:
            results.append({
                "url": url,
                "error": f"Failed to download image. HTTP status code: {response.status_code}"
            })

    return results

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
