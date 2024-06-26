from torchvision import transforms as tt
from torchvision.utils import draw_bounding_boxes
import torch

class ParkingTracker:
    def __init__(self, pklot_detector, bounding_boxes, transforms):
        pklot_detector.eval()
        self.pklot_detector = pklot_detector
        self.bounding_boxes = bounding_boxes
        self.torch_bounding_boxes = torch.tensor(bounding_boxes)
        self.transforms = transforms
    

    def crop_image(self, torch_image):
        cropped_images = []
        for bbox in self.bounding_boxes:
            xmin, ymin, xmax, ymax = bbox
            cropped_image = torch_image[:, ymin:ymax, xmin:xmax]
            cropped_image = self.transforms(cropped_image)
            cropped_images.append(cropped_image)
        return torch.stack(cropped_images)
    

    def __probabilities_to_predicts(self, probabilities):
        predicts = (probabilities > 0.5).to(torch.int)
        return predicts


    def detect(self, cropped_images):
        with torch.no_grad():
            probabilities = self.pklot_detector(cropped_images).squeeze(1)
            predicts = self.__probabilities_to_predicts(probabilities)
        return predicts
    

    def draw_bounding_boxes(self, image, predicts):
        colors = ["green" if predict.item() else "red" for predict in predicts]
        image_with_boxes = draw_bounding_boxes(image, self.torch_bounding_boxes, colors=colors, width=2, fill=True)
        return image_with_boxes

        
    @staticmethod
    def get_parking_availability(predicts):
        busy_count = sum(predicts)
        free_count = len(predicts) - busy_count
        return f"Busy: {busy_count}\nFree: {free_count}"