from utils import *


class RoadSignSegmentation:
    def __init__(self):
        self.labels_to_detect = ['TrafficSign']

        self.__load_model('data/detection/enet/model.net')
        self.__load_data('data/detection/enet/classes.txt', 'data/detection/enet/colors.txt')

    def __load_model(self, path):
        self.model = cv2.dnn.readNet(path)

    def __load_data(self, labels_path, colors_path):
        self.labels = read_file_lines(labels_path)

        colors = read_file_lines(colors_path)

        # Overwrite label colors for 0, 0, 0 if label should not be detected
        for i, l in enumerate(self.labels):
            if l not in self.labels_to_detect:
                colors[i] = '0,0,0'

        colors = [np.array(c.split(",")).astype("int") for c in colors]
        colors = np.array(colors, dtype="uint8")

        self.colors = colors
        self.labels_colors_dict = dict(zip(self.labels, self.colors))

    def transform_image_to_blob(self, image):
        # 255 pixels in grayscale
        scale_factor = 1 / 255.0

        # Shape of data that ENet was trained on
        enet_shape = (1024, 512)

        # Blob - a group of connected pixels in a binary image that share some common property (e.g. grayscale value)
        blob_image = cv2.dnn.blobFromImage(image, scale_factor, enet_shape, 0, swapRB=True, crop=False)

        return blob_image

    def get_bounding_rects_from_contours(self, contours):
        bounding_rects = []

        for contour in contours:
            bounding_rects.append(cv2.boundingRect(contour))

        return bounding_rects

    def get_mask_objects_contours(self, mask_class_map, min_object_area=1000):
        gray_mask_class_map = cv2.cvtColor(mask_class_map.astype("uint8"), cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_mask_class_map, 0, 255, cv2.THRESH_BINARY)
        detected_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        contours = []

        for contour in detected_contours:
            if cv2.contourArea(contour) > min_object_area:
                contours.append(contour)

        return contours

    def detect_signs_on_image(self, image_path, show_masked_image=False, show_legend=False):
        image = load_and_transform_image(image_path, None)
        blob_img = self.transform_image_to_blob(image)
        self.model.setInput(blob_img)
        model_output = self.model.forward()

        class_mask = np.argmax(model_output[0], axis=0)

        color_mask = self.colors[class_mask]
        color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if show_masked_image:
            self.show_masked_image(image_path, image, color_mask)

        if show_legend:
            self.show_legend()
            cv2.waitKey(0)

        contours = self.get_mask_objects_contours(color_mask)
        bounding_rects = self.get_bounding_rects_from_contours(contours)
        signs = []

        for x, y, w, h in bounding_rects:
            signs.append(image.astype("uint8")[y:y + h, x:x + w])

        return bounding_rects, signs

    def show_masked_image(self, image_path, image, mask):
        image_opacity = 0.45
        mask_opacity = 1 - image_opacity
        image_with_mask = ((image_opacity * image) + (mask_opacity * mask)).astype("uint8")
        cv2.imshow(f"Segmentation {image_path}", image_with_mask)

    def show_legend(self):
        label_color_width = 250
        label_color_height = 25

        legend = np.zeros(((len(self.labels_to_detect) * label_color_height) + label_color_height,
                           label_color_width, 3), dtype="uint8")

        i = 0
        for label, color in self.labels_colors_dict:
            if label in self.labels_to_detect:
                label_color = [int(c) for c in color]
                cv2.putText(legend, label, (5, (i * label_color_height) + 17),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(legend, (100, (i * label_color_height)),
                              (label_color_width, (i * label_color_height) + label_color_height),
                              tuple(label_color), -1)
                i = i + 1

        cv2.imshow("Labels", legend)
