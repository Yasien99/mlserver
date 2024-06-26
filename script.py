import requests

# URL of the FastAPI endpoint
url = "http://localhost:8000/predict"

# Path to the image you want to test
image_path = "test.png"

# Open the image file in binary mode
with open(image_path, "rb") as image_file:
    # Send POST request with the image file
    response = requests.post(url, files={"file": image_file})

# Print the response from the server
print(response.json())
