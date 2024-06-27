from PIL import Image
from PKLotDetector import PKLotDetector
from ParkingTracker import ParkingTracker
import pickle
import torch
import cv2
from torchvision import transforms as tt
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem

# функции для перевода иозбражения в разные типы данных
def to_torch(numpy_image):
    numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB) / 255
    return torch.from_numpy(numpy_image).permute(2, 0, 1).float()


def to_uint8(torch_image):
    torch_image *= 255
    return torch_image.to(torch.uint8)


def to_numpy(torch_image):
    torch_image = torch_image.permute(1, 2, 0).numpy()
    return cv2.cvtColor(torch_image, cv2.COLOR_BGR2RGB)

# -----------------
# Захар, это хуйня, по итогу тут получаешь parking_tracker
bounding_boxes_path = 'Usage Example/bounding_boxes.pkl'
with open(bounding_boxes_path, 'rb') as f:
    bounding_boxes = pickle.load(f)
weights_path = "pklot_detector.pth"
pklot_detector = PKLotDetector()
pklot_detector.load_state_dict(torch.load(weights_path, map_location='cpu'))
transforms = tt.Resize((100, 100))
parking_tracker = ParkingTracker(pklot_detector, bounding_boxes, transforms)
# --------------


# путь к изображение
image_path = r"C:\Users\Azaki\Desktop\studie\ParkingTracker\Usage Example\image_to_annotate.jpg"

# открывем изображение по пути
image = cv2.imread(image_path)

# переводим в нужный тип данных
torch_image = to_torch(image)

# режем фотки на парковочные места
cropped_images = parking_tracker.crop_image(torch_image)

# получаем предасказания, где свободно а где занято
predicts = parking_tracker.detect(cropped_images)

# переводим в нужный тип данных для рисования коробок
torch_image = to_uint8(torch_image)

# рисуем коробки
drawed_image = parking_tracker.draw_bounding_boxes(torch_image, predicts)

# переводим в нужный тип данных для вывода изображдения
output_image = to_numpy(drawed_image)
output_image = Image.fromarray(output_image)




# # получаем 2 числа, сколько свободных, сколько занятый(это просто пример использования функции, для этого файла не важно)
availability = parking_tracker.get_parking_availability(predicts)

# #  пример перервода предсказания в стандартный list, не знаю зачем, но может пригодится. А так мне кажется что работу с предсказаниями лучше организовать в классе ParkingTracker, ну сам думай, если че я могу добавить какието функции в parkingtracker просто
#predicts_list = [i.item() for i in predicts]




class MyInterfaceApp(App):
    def build(self):
        main_place = TabbedPanel(do_default_tab=False)
        first_window = TabbedPanelItem(text='Count_of_free_parking_places')
        first_window_box = BoxLayout()


        main_lable = Label(text=availability)
        first_window_box.add_widget(main_lable)
        first_window.add_widget(first_window_box)

        main_place.add_widget(first_window)

        return main_place


if __name__ == "__main__":
    MyInterfaceApp().run()
