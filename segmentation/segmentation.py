import cv2
import numpy as np
from detection.bounding_box import BoundingBox
from utils.utils import read_file_lines


class SemanticSegmentationMask:
    def __init__(self, color_mask, label_id, label, label_name, label_color, masked_image_opacity):
        self.label_id = label_id
        self.label = label
        self.label_name = label_name
        self.label_color = label_color
        self.masked_image_opacity = masked_image_opacity
        self.mask = np.zeros(color_mask.shape, dtype="uint8")
        self.mask[np.where((color_mask == label_color).all(axis=2))] = label_color
        self.label_pixel_coverage_percent = self.get_label_pixel_coverage_percent()

    def get_bounding_boxes(self, detection_min_object_area):
        bounding_boxes: [BoundingBox] = []

        gray_mask_class_map = cv2.cvtColor(self.mask.astype("uint8"), cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_mask_class_map, 0, 255, cv2.THRESH_BINARY)
        detected_contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]

        for contour in detected_contours:
            if cv2.contourArea(contour) > detection_min_object_area:
                (x, y, w, h) = cv2.boundingRect(contour)
                bounding_boxes.append(BoundingBox(x, y, x + w, y + h, self.label_id, self.label_name,
                                                  tuple((int(x) for x in self.label_color))))

        return bounding_boxes

    def get_label_pixel_coverage_percent(self):
        mask_pixels = self.mask.shape[1] * self.mask.shape[0]
        pixel_number = np.count_nonzero(np.all(self.mask == self.label_color, axis=2))

        return pixel_number/mask_pixels

    def draw_mask_on_image(self, image):
        mask_opacity = 1 - self.masked_image_opacity
        return ((self.masked_image_opacity * image) + (mask_opacity * self.mask)).astype("uint8")


class SemanticSegmentation:
    def __init__(self, config):
        self.labels_to_detect = config['labels_to_detect']
        self.detection_min_object_area = config['detection_min_object_area']
        self.masked_image_opacity = config['masked_image_opacity']

        self.__load_model(config['load_model_path'])
        self.__load_data(config['labels_path'], config['label_names_path'], config['colors_path'])

    def __load_model(self, path):
        self.model = cv2.dnn.readNet(path)

        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def __load_data(self, labels_path, label_names_path, colors_path):
        self.labels = read_file_lines(labels_path)
        self.label_names = read_file_lines(label_names_path)

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

    def split_mask_by_objects(self, color_mask):
        masks: [SemanticSegmentationMask] = []
        bounding_boxes: [BoundingBox] = []

        for label in self.labels_to_detect:
            label_id = self.labels.index(label)
            mask = SemanticSegmentationMask(color_mask, label_id, label, self.label_names[label_id],
                                            self.labels_colors_dict[label], self.masked_image_opacity)
            masks.append(mask)
            bounding_boxes += mask.get_bounding_boxes(self.detection_min_object_area)

        return masks, bounding_boxes

    def detect_objects_on_image(self, image, show_legend=False):
        blob_img = self.transform_image_to_blob(image)
        self.model.setInput(blob_img)
        model_output = self.model.forward()

        class_mask = np.argmax(model_output[0], axis=0)

        color_mask = self.colors[class_mask]
        color_mask = cv2.resize(color_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        if show_legend:
            self.show_legend()
            cv2.waitKey(0)

        return self.split_mask_by_objects(color_mask)

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
