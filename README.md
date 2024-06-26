
---

# ParkingTracker

ParkingTracker is a project designed for detecting parking space occupancy using computer vision techniques. It includes tools for annotation, cropping, dataset management, and a deep learning model for occupancy prediction.

## Project Structure

### Files

1. **crop_by_coco.ipynb**: This serve for crop and save images based on COCO annotations. It processes multiple datasets and saves cropped images categorized by occupancy.

2. **crop.ipynb**: Implements image cropping based on bounding boxes from various datasets.

3. **PKLotDetector.py**: Defines a PyTorch neural network model (PKLotDetector) for binary classification of parking spaces. It includes convolutional and fully connected layers for occupancy prediction.

4. **ParkingTracker.py**: ParkingTracker contains functions that analyze parking data and visualize made analyze. It uses PKLotDetector at a high level and supports dataset augmentation and preprocessing. 

5. **Parking_detection_usage_example.ipynb**: Offers usage examples for ParkingTracker, demonstrating how to create bounding boxes, train models, and use the detector for inference.

### Directories

- **Data**: Contains various parking datasets used for training and testing.
- **CROPPED**: Stores cropped images categorized by occupancy status (busy or free).

## Usage

1. To use the Parking Tracker, you first need to create bounding boxes, which is a quick and straightforward process. This procedure needs to be done once initially, and then the model can autonomously operate as long as the camera remains static (bounding boxes must accurately match the actual parking spot locations for proper functioning). to create bounding boxes: Follow inctructions from bounding_boxes_creation.py file from Usage Example folder with LabelImg to define bounding boxes for each parking spot.
2. Place each parking spot within its respective bounding box. This step ensures the neural network knows where to locate vehicles.<img width="1440" alt="annotate_bounding_boxes" src="https://github.com/osipgas/ParkingTracker/assets/115102730/be004f64-2301-4cd7-b462-a9d28817f64b">
3. Once bounding boxes are created, proceed to the parking_tracker_usage_example.py file and follow the provided instructions.
