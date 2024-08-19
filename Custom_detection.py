from ultralytics import YOLO

# Create an instance of the YOLOv8 model
model = YOLO(weights="runs_new\\detect\\train3\\weights\\best.pt")


# Process images with the model
results = model("images.jpeg")

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # Display to screen or save to disk
