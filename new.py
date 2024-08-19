from ultralytics import YOLO

# Create an instance of the YOLOv8 model
model = YOLO(weights=r"C:/Users/sayli/OneDrive/Documents/AIES_MINIPROJ/runs_new/detect/train3/weights/best.pt")

# Process images with the model
results = model("images.jpeg")

# Process results list
for result in results.pred:
    boxes = result.xyxy  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.conf  # Confidence scores for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # Display to screen or save to disk
