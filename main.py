import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os


app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = os.path.join(os.getcwd(), 'best v3(100).pt')
logger.info(f"Loading model from path: {model_path}")

if not os.path.exists(model_path):
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = YOLO(model_path)



@app.post("/v1/api/watermeter/detect/reading/value")
async def detect_water_meter(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        logger.info(f"Received file of size: {len(image_data)} bytes")

        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB') 
            logger.info("Image successfully opened and verified.")
        except UnidentifiedImageError:
            logger.error("Failed to identify image.")
            raise HTTPException(status_code=400, detail="Invalid image file.")

        img_array = np.array(image)
        logger.info(f"Image converted to array of shape: {img_array.shape}")

        if img_array.size == 0:
            logger.error("Image data is empty.")
            raise HTTPException(status_code=400, detail="Image data is empty.")

        logger.info("Running inference on the image...")
        results = model.predict(source=img_array, conf=0.25, imgsz=640)
        logger.info(f"Raw model prediction results: {results}")

        detected_numbers = []
        for result in results:
            logger.info(f"Processing result: {result}")
            for box in result.boxes:
                label_index = int(box.cls[0].item())  # Get the predicted class index
                label = result.names[label_index]  # Get the label (e.g., a number or text)
                x1, _, _, _ = box.xyxy[0].tolist()  # Extract bounding box coordinates

                logger.info(f"Detected label: {label} at position: {x1}")

                if label.isdigit():  # Check if the label is a number
                    detected_numbers.append((x1, label))

        detected_numbers.sort(key=lambda x: x[0])
        sorted_numbers = [num for _, num in detected_numbers]
        detected_values_str = ''.join(sorted_numbers)  # Combine numbers into a single string

        logger.info(f"Detected values: {detected_values_str}")

        
        return JSONResponse(
            content={
                "status": "success",
                "detected_values": detected_values_str if sorted_numbers else "No numeric values detected."
            }
        )

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# If running locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, timeout_keep_alive=120)
    