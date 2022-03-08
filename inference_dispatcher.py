import time

from bounding_box import BoundingBox
from classification import RoadSignClassification
from detection_FasterRCNN import RoadSignFasterRCNNDetection
from segmentation import SemanticSegmentation, SemanticSegmentationMask


class InferenceDispatcher:
    def __init__(self, detection_config, classification_config, segmentation_config):
        self.detection = RoadSignFasterRCNNDetection(detection_config, mode='inference') \
            if detection_config else None

        self.detection_threshold = detection_config['detection_threshold'] if detection_config else None

        self.classification = RoadSignClassification(classification_config, mode='inference') \
            if classification_config else None

        self.segmentation = SemanticSegmentation(segmentation_config) if segmentation_config else None

    def dispatch(self, image):
        detected_images = [image]
        detected_bounding_boxes: [BoundingBox] = []
        segmentation_bounding_boxes: [BoundingBox] = []
        masks: [SemanticSegmentationMask] = []

        start = time.time()

        if self.detection:
            detected_bounding_boxes = self.detection.predict_bounding_boxes(image, self.detection_threshold)
            detected_images = [bounding_box.image for bounding_box in detected_bounding_boxes]

        if self.classification:
            classified_labels = self.classification.model_predict_data(detected_images)
            for (index, bounding_box) in enumerate(detected_bounding_boxes):
                (classified_label_id, classified_label_name) = classified_labels[index]
                bounding_box.label_id = classified_label_id
                bounding_box.label_name = classified_label_name

        if self.segmentation:
            masks, segmentation_bounding_boxes = self.segmentation.detect_objects_on_image(image)

        end = time.time()
        elapsed = end - start

        print(f"Inference took {elapsed:.3f} seconds.")

        return detected_bounding_boxes, segmentation_bounding_boxes, masks, elapsed
