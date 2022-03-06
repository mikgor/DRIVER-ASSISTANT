import os
import sys

import cv2
import yaml
import numpy as np
import imgaug as ia

from classification import RoadSignClassification
from detection_FasterRCNN import RoadSignFasterRCNNDetection
from inference_dispatcher import InferenceDispatcher
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


def input_from_options(title, options, selected_options, allow_multiple=False):
    def selected_options_are_correct():
        if selected_options is None or len(selected_options) == 0:
            return False

        if not allow_multiple and len(selected_options) > 1:
            return False

        for selected_option in selected_options:
            if selected_option not in option_ids:
                return False
            print(options[selected_option - 1][1])

        return True

    option_ids = [idx for idx, option in options]

    if type(selected_options) is not list and selected_options is not None:
        selected_options = [int(option.strip()) for option in selected_options.split(',') if option.isdigit()] \
            if allow_multiple else [int(selected_options)]

    if selected_options_are_correct():
        return selected_options if allow_multiple else selected_options[0]

    while True:
        print(f'Choose {title}. Options:')
        for (idx, option) in options:
            print(f'[{idx}] - {option}')

        choice_message = 'Choice: ' + ('(multiple, separated by ,) ' if allow_multiple else '(single) ')
        selected_options = [int(option.strip()) for option in input(choice_message).split(',') if option.isdigit()]

        if selected_options_are_correct():
            return selected_options if allow_multiple else selected_options[0]


def display_menu(config):
    MODE_INFERENCE_IMAGES = 1
    MODE_INFERENCE_VIDEOS = 2
    MODE_TRAIN = 3

    INFERENCE_DETECTION = 1
    INFERENCE_CLASSIFICATION = 2
    INFERENCE_SEMANTIC_SEGMENTATION = 3

    TRAIN_DETECTION = 1
    TRAIN_CLASSIFICATION = 2

    startup_config = config['startup']
    inference_config = config['inference']

    mode_options = [(MODE_INFERENCE_IMAGES, 'Inference (images)'),
                    (MODE_INFERENCE_VIDEOS, 'Inference (videos)'),
                    (MODE_TRAIN, 'Train')]
    mode_selected_option = input_from_options('Model mode', mode_options, startup_config['mode_selected_option'])

    if mode_selected_option != MODE_TRAIN:
        inference_options = [(INFERENCE_DETECTION, 'Detection'),
                             (INFERENCE_CLASSIFICATION, 'Classification'),
                             (INFERENCE_SEMANTIC_SEGMENTATION, 'Semantic segmentation')]
        inference_selected_options = input_from_options('Inference function', inference_options,
                                                        startup_config['function_selected_options'], True)

        inference_dispatcher = InferenceDispatcher(
            config['detection'] if INFERENCE_DETECTION in inference_selected_options else None,
            config['classification'] if INFERENCE_CLASSIFICATION in inference_selected_options else None,
            config['segmentation'] if INFERENCE_SEMANTIC_SEGMENTATION in inference_selected_options else None,)

        if mode_selected_option == MODE_INFERENCE_IMAGES:
            images_path = inference_config['images_path']

            for image_name in os.listdir(images_path):
                image_path = '{}/{}'.format(images_path, image_name)
                image = load_and_transform_image(image_path, None)

                image, detected_bounding_boxes, detected_images, detected_label_ids, detected_scores, \
                    classified_labels, elapsed = inference_dispatcher.dispatch(image)

                image = draw_rectangles_and_text_on_image_from_bounding_boxes(
                    image, detected_bounding_boxes, classified_labels)

                cv2.imshow(image_name, image)

        elif mode_selected_option == MODE_INFERENCE_VIDEOS:
            videos_path = inference_config['videos_path']
            videos_output_path = inference_config['videos_output_path']
            for video_name in os.listdir(videos_path):
                if '.csv' in video_name:
                    continue

                video_path = '{}/{}'.format(videos_path, video_name)
                video_output_path = '{}/{}'.format(videos_output_path, video_name)
                df = get_video_detections(video_output_path)
                start_frame_id = int(df.iloc[-1]['FrameId'])+1 if df.size > 0 else 0
                frame_id = start_frame_id
                time_elapsed = 0

                cap = cv2.VideoCapture(video_path)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                success = True

                while success and frame_id < frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id+frame_id)
                    success, frame = cap.read()

                    frame, detected_bounding_boxes, detected_images, detected_label_ids, detected_scores, \
                        classified_labels, elapsed = inference_dispatcher.dispatch(frame)

                    time_elapsed += elapsed
                    avg_sec_per_frame = time_elapsed / (frame_id+1 - start_frame_id)

                    print(f"Processed {(frame_id/frames):.3f}% ({frame_id} / {frames}). Time left: "
                          f"{((avg_sec_per_frame*(frames-frame_id))/60):.3f} minutes.")

                    save_video_frame_detections(
                        frame_id, detected_bounding_boxes, detected_label_ids, detected_scores, df, video_output_path)

                    frame_id += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    success, frame = cap.read()

                play_video_with_labels(cap, df, config['classification'])

        cv2.waitKey(0)

    elif mode_selected_option == MODE_TRAIN:
        train_options = [(TRAIN_DETECTION, 'Detection'), (TRAIN_CLASSIFICATION, 'Classification')]
        train_selected_option = input_from_options('Train function', train_options,
                                                   startup_config['function_selected_options'])

        if train_selected_option == TRAIN_DETECTION:
            _ = RoadSignFasterRCNNDetection(config['detection'], mode='train')

        elif train_selected_option == TRAIN_CLASSIFICATION:
            _ = RoadSignClassification(config['classification'], mode='train')


if __name__ == '__main__':
    config = load_configuration()

    seed = config['startup']['seed']
    ia.seed(seed)
    np.random.seed(seed)

    display_menu(config)
