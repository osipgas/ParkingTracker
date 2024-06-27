# ParkingTracker

Welcome to ParkingTracker, computer vision project designed to detect parking occupancy using computer vision techniques.


## Key Features

- **High Accuracy**: Achieved an impressive 98.5% accuracy on a diverse range of datasets, ensuring reliable performance across various environments.
- **Real-time Detection**: Processes images and video streams in real-time, providing instant feedback on parking spaces availability.
- **Scalability**: Since the PKLotDetector operates swiftly, it is well-suited for managing all parkings, including large ones.


## Project Structure

### Files

1. **crop_by_coco.ipynb**: Serves for crop and save images based on COCO annotations. It processes multiple datasets and saves cropped images categorized by occupancy.

2. **crop.ipynb**: Implements image cropping based on bounding boxes from various datasets.

3. **PKLotDetector.py**: Defines a PyTorch neural network model (PKLotDetector) for binary classification of parking spaces. It includes convolutional and fully connected layers for occupancy prediction.

4. **ParkingTracker.py**: ParkingTracker contains functions that analyze parking data and visualize made analyze. It uses PKLotDetector at a high level and supports dataset augmentation and preprocessing. 

5. **Parking_detection_usage_example.ipynb**: Offers usage examples for ParkingTracker, demonstrating how to create bounding boxes, train models, and use the detector for inference.

### Directories

- **Data**: Contains various parking datasets used for training and testing. To achieve good accuracy on various data, I had to use multiple datasets.
- **CROPPED**: Every dataset from "Data" folder includes "CROPPED" folder that stores cropped images categorized by occupancy status (busy or free). I'am creating this folders in "crop.ipynb" and "crop_by_coco.ipynb" files.

## Usage

1. To use the Parking Tracker, you first need to create bounding boxes, which is a quick and straightforward process. This procedure needs to be done once initially, and then the model can autonomously operate as long as the camera remains static (bounding boxes must accurately match the actual parking spot locations for proper functioning). to create bounding boxes: Follow inctructions from bounding_boxes_creation.py file from Usage Example folder with LabelImg to define bounding boxes for each parking spot.
2. Place each parking spot within its respective bounding box. This step ensures the neural network knows where to locate vehicles.<img width="1440" alt="annotate_bounding_boxes" src="https://github.com/osipgas/ParkingTracker/assets/115102730/be004f64-2301-4cd7-b462-a9d28817f64b">
To download and open LabelImg run this commands in terminal:

    git clone https://github.com/tzutalin/labelImg

    cd labelImg

    make qt5py3

    python labelImg.py


4. Once bounding boxes are created, proceed to the parking_tracker_usage_example.py file and follow the provided instructions. With ParkingTracker you can get predicts, quantity of free and occupied parking lots, image with visualated predicts.



The following image shows the result of ParkingTracker work:![output](https://github.com/osipgas/ParkingTracker/assets/115102730/bf21654b-4e8e-4dec-943a-39718226e273)




And here's the video example:

https://github.com/osipgas/ParkingTracker/assets/115102730/3197c549-7ba6-498e-8969-4114b4f3866b




