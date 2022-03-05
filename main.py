import os
import sys
import time

import cv2
import yaml
import numpy as np
import imgaug as ia

from classification import RoadSignClassification
from detection_FasterRCNN import RoadSignFasterRCNNDetection
from segmentation import SemanticSegmentation
from utils import load_and_transform_image, draw_rectangles_and_text_on_image_from_bounding_boxes, get_video_detections, \
    save_video_frame_detections, play_video_with_labels


def load_configuration():
    config_path = 'configurations/_config.yaml'

    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    with open(config_path, 'r') as stream:
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
        inference_function_options = [(1, 'Detection & Classification (images) + Semantic segmentation (optional)'),
                                      (2, 'Detection & Classification (videos) + Semantic segmentation (optional)'),
                                      (3, 'Classification'), (4, 'Semantic segmentation')]
        inference_function_selected_option = input_from_options('Inference function', inference_function_options,
                                                                startup_config['function_selected_option'])

        if inference_function_selected_option == 1 or inference_function_selected_option == 2:
            optional_semantic_segmentation_options = [(1, 'Yes'), (2, 'No')]
            optional_semantic_segmentation_selected_option = \
                input_from_options('Apply semantic segmentation', optional_semantic_segmentation_options,
                                   startup_config['detection_and_classification_optional_semantic_segmentation'])

            if optional_semantic_segmentation_selected_option == 1:
                segmentation = SemanticSegmentation(config['segmentation'])

        if inference_function_selected_option == 1:
            detection = RoadSignFasterRCNNDetection(config['detection'], mode='inference')
            classification = RoadSignClassification(config['classification'], mode='inference')

            images_folder_path = startup_config['detection_and_classification_path']
            for image_name in os.listdir(images_folder_path):
                image_path = '{}/{}'.format(images_folder_path, image_name)
                image = load_and_transform_image(image_path, None)

                bounding_boxes, signs, label_ids, scores = detection.predict_boxes_and_images(
                    image_path, detection_threshold=config['detection']['detection_threshold'])

                predicted_labels = classification.model_predict_data(signs)

                if optional_semantic_segmentation_selected_option == 1:
                    masks = segmentation.detect_objects_on_image(image)
                    for mask in masks:
                        image = mask.draw_mask_bounding_boxes(image)

                image = \
                    draw_rectangles_and_text_on_image_from_bounding_boxes(image, bounding_boxes, predicted_labels)

                cv2.imshow(image_name, image)

        elif inference_function_selected_option == 2:
            detection = RoadSignFasterRCNNDetection(config['detection'], mode='inference')
            classification = RoadSignClassification(config['classification'], mode='inference')

            videos_folder_path = startup_config['detection_and_classification_videos_path']
            for video in os.listdir(videos_folder_path):
                if '.csv' in video:
                    continue

                video_path = '{}/{}'.format(videos_folder_path, video)
                df = get_video_detections(video_path)
                start_frame_id = int(df.iloc[-1]['FrameId'])+1 if df.size > 0 else 0
                frame_id = start_frame_id
                time_elapsed = 0

                cap = cv2.VideoCapture(video_path)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                success = True

                while success and frame_id < frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id+frame_id)
                    success, frame = cap.read()

                    start = time.time()

                    bounding_boxes, signs, _, scores = detection.predict_boxes_and_images(
                        frame, detection_threshold=config['detection']['detection_threshold'])

                    predicted_labels = classification.model_predict_data(signs, return_label_name=False)

                    end = time.time()
                    elapsed = end - start
                    time_elapsed += elapsed
                    avg_sec_per_frame = time_elapsed / (frame_id+1 - start_frame_id)

                    print(f"Frame {frame_id} took {elapsed:.3f} seconds. Processed {(frame_id/frames):.3f}% "
                          f"({frame_id} / {frames}). Time left: "
                          f"{((avg_sec_per_frame*(frames-frame_id))/60):.3f} minutes.")

                    save_video_frame_detections(frame_id, bounding_boxes, predicted_labels, scores, df, video_path)

                    frame_id += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    success, frame = cap.read()

                play_video_with_labels(cap, df, config['classification'],
                                       segmentation if optional_semantic_segmentation_selected_option == 1 else None)

        elif inference_function_selected_option == 3:
            classification = RoadSignClassification(config['classification'], mode='inference')
            images_folder_path = startup_config['classification_path']
            images = []

            for sign_code in os.listdir(images_folder_path):
                for sign_image_name in os.listdir('{}/{}'.format(images_folder_path, sign_code)):
                    image_path = '{}/{}/{}'.format(images_folder_path, sign_code, sign_image_name)
                    images.append(load_and_transform_image(image_path, None))

            _ = classification.model_predict_data(images, show_images=True)

        elif inference_function_selected_option == 4:
            segmentation = SemanticSegmentation(config['segmentation'])
            images_folder_path = startup_config['semantic_segmentation_path']

            for image_name in os.listdir(images_folder_path):
                image_path = '{}/{}'.format(images_folder_path, image_name)
                image = load_and_transform_image(image_path, None)

                masks = segmentation.detect_objects_on_image(image)
                for mask in masks:
                    image = mask.draw_mask_bounding_boxes(image)

                cv2.imshow(image_name, image)

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
