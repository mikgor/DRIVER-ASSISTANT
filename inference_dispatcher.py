import time

from classification import RoadSignClassification
from detection_FasterRCNN import RoadSignFasterRCNNDetection
from segmentation import SemanticSegmentation


class InferenceDispatcher:
    def __init__(self, detection_config, classification_config, segmentation_config):
        self.detection = RoadSignFasterRCNNDetection(detection_config, mode='inference') \
            if detection_config else None

        self.detection_threshold = detection_config['detection_threshold'] if detection_config else None

        self.classification = RoadSignClassification(classification_config, mode='inference') \
            if classification_config else None

        self.segmentation = SemanticSegmentation(segmentation_config) if segmentation_config else None

    def dispatch(self, image):
        detected_bounding_boxes = []
        detected_images = []
        detected_label_ids = []
        detected_scores = []
        classified_labels = []

        start = time.time()

        if self.detection:
            detected_bounding_boxes, detected_images, detected_label_ids, detected_scores = \
                self.detection.predict_boxes_and_images(image, detection_threshold=self.detection_threshold)

        if self.classification:
            classified_labels = self.classification.model_predict_data(detected_images)

        if self.segmentation:
            masks = self.segmentation.detect_objects_on_image(image)
            for mask in masks:
                image = mask.draw_mask_bounding_boxes(image)

        end = time.time()
        elapsed = end - start

        print(f"Inference took {elapsed:.3f} seconds.")

        return image, detected_bounding_boxes, detected_images, detected_label_ids, detected_scores, classified_labels, elapsed
