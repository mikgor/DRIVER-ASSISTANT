from detection_FasterRCNN import RoadSignFasterRCNNDetection
from classification import RoadSignClassification
from utils import *

# detection = RoadSignFasterRCNNDetection()
# detection.train_model()

detection = RoadSignFasterRCNNDetection(mode='inference', model_path='data/detection/models/fasterrcnn/model4.pth')

images_folder_path = 'data/detection/images/'

rsc = RoadSignClassification(test_data_path=None)


for image_name in os.listdir(images_folder_path):
    image_path = '{}/{}'.format(images_folder_path, image_name)
    image = load_and_transform_image(image_path, None)

    bounding_boxes, signs = detection.predict_boxes_and_images(image_path)
    predicted_labels = rsc.model_predict_data(signs)

    for (index, bounding_box) in enumerate(bounding_boxes):
        image = draw_rectangle_on_image_from_bounding_box(image, bounding_box)
        image = draw_text_under_object_on_image_from_bounding_box(image, bounding_box, predicted_labels[index])

    cv2.imshow(image_name, image)

cv2.waitKey(0)
