import os

import cv2
import yaml
import numpy as np
import imgaug as ia

from classification import RoadSignClassification
from detection_FasterRCNN import RoadSignFasterRCNNDetection
from segmentation import RoadSignSegmentation
from utils import load_and_transform_image, draw_rectangles_and_text_on_image_from_bounding_boxes


def load_configuration():
    with open('configurations/_config.yaml', 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


def input_from_options(title, options, selected_option):
    option_ids = [idx for idx, option in options]

    if selected_option in option_ids:
        print(options[selected_option - 1][1])
        return selected_option

    while True:
        print(f'Choose {title}. Options:')
        for (idx, option) in options:
            print(f'[{idx}] - {option}')

        selected_option = int(input('Choice: '))
        if selected_option in option_ids:
            print(options[selected_option-1][1])
            return selected_option


def display_menu(config):
    startup_config = config['startup']

    mode_options = [(1, 'Inference'), (2, 'Train')]
    mode_selected_option = input_from_options('Model mode', mode_options, startup_config['mode_selected_option'])

    if mode_selected_option == 1:
        inference_function_options = [(1, 'Detection & Classification'),
                                      (2, 'Classification'), (3, 'Semantic segmentation')]
        inference_function_selected_option = input_from_options('Inference function', inference_function_options,
                                                                startup_config['function_selected_option'])

        if inference_function_selected_option == 1:
            detection = RoadSignFasterRCNNDetection(config['detection'], mode='inference')
            classification = RoadSignClassification(config['classification'], mode='inference')

            images_folder_path = startup_config['detection_and_classification_path']
            for image_name in os.listdir(images_folder_path):
                image_path = '{}/{}'.format(images_folder_path, image_name)
                image = load_and_transform_image(image_path, None)

                bounding_boxes, signs = detection.predict_boxes_and_images(
                    image_path, detection_threshold=config['detection']['detection_threshold'])

                predicted_labels = classification.model_predict_data(signs)

                image = \
                    draw_rectangles_and_text_on_image_from_bounding_boxes(image, bounding_boxes, predicted_labels)

                cv2.imshow(image_name, image)

        elif inference_function_selected_option == 2:
            classification = RoadSignClassification(config['classification'], mode='inference')
            images_folder_path = startup_config['classification_path']
            images = []

            for sign_code in os.listdir(images_folder_path):
                for sign_image_name in os.listdir('{}/{}'.format(images_folder_path, sign_code)):
                    image_path = '{}/{}/{}'.format(images_folder_path, sign_code, sign_image_name)
                    images.append(load_and_transform_image(image_path, None))

            _ = classification.model_predict_data(images, show_images=True)

        elif inference_function_selected_option == 3:
            segmentation = RoadSignSegmentation()
            images_folder_path = startup_config['semantic_segmentation_path']

            for image_name in os.listdir(images_folder_path):
                image_path = '{}/{}'.format(images_folder_path, image_name)

                _, _ = segmentation.detect_signs_on_image(image_path=image_path, show_masked_image=True)

        cv2.waitKey(0)

    elif mode_selected_option == 2:
        train_function_options = [(1, 'Detection'), (2, 'Classification')]
        train_function_selected_option = input_from_options('Inference function', train_function_options,
                                                            startup_config['function_selected_option'])

        if train_function_selected_option == 1:
            _ = RoadSignFasterRCNNDetection(config['detection'], mode='train')

        elif train_function_selected_option == 2:
            _ = RoadSignClassification(config['classification'], mode='train')


if __name__ == '__main__':
    config = load_configuration()

    seed = config['startup']['seed']
    ia.seed(seed)
    np.random.seed(seed)

    display_menu(config)
