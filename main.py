from classification import RoadSignClassification
from detection import RoadSignDetection
from utils import *

images_folder_path = 'data/detection/images/'
rsd = RoadSignDetection()

# rsc = RoadSignClassification()
# print('Making predictions on test data...')
# rsc.model_predict_test_data(show_images=True)
rsc = RoadSignClassification(test_data_path=None)


for image_name in os.listdir(images_folder_path):
    image_path = '{}/{}'.format(images_folder_path, image_name)
    image = load_and_transform_image(image_path, None)

    bounding_rects, signs = rsd.detect_signs_on_image(image_path=image_path)
    predicted_labels = rsc.model_predict_data(signs)

    for (index, bounding_rect) in enumerate(bounding_rects):
        image = draw_rectangle_on_image(image, bounding_rect)
        image = draw_text_under_object_on_image(image, bounding_rect, predicted_labels[index])

    cv2.imshow(image_name, image)

cv2.waitKey(0)
