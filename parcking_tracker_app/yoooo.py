from PIL import Image
from ParkingTracker.PKLotDetector import PKLotDetector
from ParkingTracker.ParkingTracker import ParkingTracker
import pickle
import torch
import cv2
from torchvision import transforms as tt
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.image import Image as Im
from kivy.graphics.texture import Texture
from kivy.clock import Clock

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
bounding_boxes_path = r'C:\Users\Azaki\Desktop\studie\ParkingTracker\Usage Example\bounding_boxes.pkl'
with open(bounding_boxes_path, 'rb') as f:
    bounding_boxes = pickle.load(f)
weights_path = r"C:\Users\Azaki\Desktop\studie\ParkingTracker\pklot_detector.pth"
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
        main_place = TabbedPanel(do_default_tab=False, tab_pos='top_mid', background_color=(60/255, 179/255, 113/255))
        first_window = TabbedPanelItem(text='Parcking', background_color=(60/255, 179/255, 113/255))
        second_window = TabbedPanelItem(text='Video', background_color=(60/255, 179/255, 113/255))
        first_window_box = BoxLayout()
        second_window_box = BoxLayout()

        main_lable = Label(text=availability, font_size='40sp')
        first_window_box.add_widget(main_lable)
        first_window.add_widget(first_window_box)

        self.img = Im()
        second_window_box.add_widget(self.img)
        second_window.add_widget(second_window_box)

        main_place.add_widget(first_window)
        main_place.add_widget(second_window)
        self.img_path = image_path
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return main_place

    def update(self, dt):
        frame = cv2.imread(self.img_path)
        if frame is not None    :
            # Convert the frame to texture
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = texture


if __name__ == "__main__":
    MyInterfaceApp().run()
