import os
import sys
import time

import cv2
import yaml
import numpy as np
import imgaug as ia
import pandas as pd

from classification import RoadSignClassification
from detection import RoadSignDetection
from image_labeling import label_image
from inference_dispatcher import InferenceDispatcher
from utils import load_and_transform_image, get_video_dfs, draw_bounding_boxes_on_image, save_video_frame_dfs, \
    add_prefix_before_file_extension, read_file_lines, get_video_df_frame_bounding_boxes, get_gtsrb_df


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
    MODE_INFERENCE_LIVE = 3
    MODE_TRAIN = 4
    MODE_PLAY_LABELED_VIDEOS = 5
    MODE_LABEL_IMAGES = 6

    INFERENCE_DETECTION = 1
    INFERENCE_CLASSIFICATION = 2
    INFERENCE_SEMANTIC_SEGMENTATION = 3

    TRAIN_DETECTION = 1
    TRAIN_CLASSIFICATION = 2

    startup_config = config['startup']
    inference_config = config['inference']
    inference_videos_config = inference_config['videos']
    image_labeling_config = config['image_labeling']

    mode_options = [(MODE_INFERENCE_IMAGES, 'Inference (images)'),
                    (MODE_INFERENCE_VIDEOS, 'Inference (videos)'),
                    (MODE_INFERENCE_LIVE, 'Inference (live)'),
                    (MODE_TRAIN, 'Train'),
                    (MODE_PLAY_LABELED_VIDEOS, 'Apply labels to videos and play'),
                    (MODE_LABEL_IMAGES, 'Label images')]
    mode_selected_option = input_from_options('Model mode', mode_options, startup_config['mode_selected_option'])

    if mode_selected_option != MODE_TRAIN and mode_selected_option != MODE_PLAY_LABELED_VIDEOS:
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

                bounding_boxes, segmentation_bounding_boxes, masks, _ = inference_dispatcher.dispatch(image)

                image = draw_bounding_boxes_on_image(image, bounding_boxes + segmentation_bounding_boxes)

                cv2.imshow(image_name, image)

        elif mode_selected_option == MODE_INFERENCE_VIDEOS:
            videos_path = inference_videos_config['path']
            videos_output_path = inference_videos_config['output_path']
            for video_name in os.listdir(videos_path):
                video_path = '{}/{}'.format(videos_path, video_name)
                video_output_path = '{}/{}'.format(videos_output_path, video_name)
                detections_df, segmentations_df = get_video_dfs(video_output_path)
                start_frame_id = max([int(detections_df.iloc[-1]['FrameId'])+1 if detections_df.size > 0 else 0,
                                     int(segmentations_df.iloc[-1]['FrameId'])+1 if segmentations_df.size > 0 else 0])
                frame_id = start_frame_id
                time_elapsed = 0

                cap = cv2.VideoCapture(video_path)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                success = True

                out = cv2.VideoWriter(add_prefix_before_file_extension(video_output_path, 'labeled'),
                                      cv2.VideoWriter_fourcc(*'mp4v'), fps, size)  \
                    if inference_videos_config['save_labeled'] else None

                while success and frame_id < frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id+frame_id)
                    success, frame = cap.read()

                    detected_bounding_boxes, segmentation_bounding_boxes, masks, elapsed = \
                        inference_dispatcher.dispatch(frame)

                    time_elapsed += elapsed
                    avg_sec_per_frame = time_elapsed / (frame_id+1 - start_frame_id)

                    print(f"Processed {(frame_id/frames):.3f}% ({frame_id} / {frames}). Time left: "
                          f"{((avg_sec_per_frame*(frames-frame_id))/60):.3f} minutes.")

                    save_video_frame_dfs(frame_id, detected_bounding_boxes, detections_df,
                                         segmentation_bounding_boxes, segmentations_df, video_output_path)

                    if out is not None or inference_videos_config['show_labeled_frame']:
                        frame = \
                            draw_bounding_boxes_on_image(frame, detected_bounding_boxes + segmentation_bounding_boxes)

                        if out is not None:
                            out.write(frame)

                        if inference_videos_config['show_labeled_frame']:
                            cv2.imshow(video_name, frame)
                            cv2.waitKey(1)

                    frame_id += 1

                cap.release()
                if out is not None:
                    out.release()

        elif mode_selected_option == MODE_INFERENCE_LIVE:
            video_name = f'{time.time()}.mp4'
            video_output_path = '{}/{}'.format(inference_videos_config['output_path'], video_name)
            detections_df, segmentations_df = get_video_dfs(video_output_path)
            cap = cv2.VideoCapture(0)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            out = cv2.VideoWriter(add_prefix_before_file_extension(video_output_path, 'labeled'),
                                  cv2.VideoWriter_fourcc(*'mp4v'), fps, size) \
                if inference_videos_config['save_labeled'] else None

            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame.")
                    break

                detected_bounding_boxes, segmentation_bounding_boxes, masks, _ = \
                    inference_dispatcher.dispatch(frame)

                save_video_frame_dfs(frame_id, detected_bounding_boxes, detections_df,
                                     segmentation_bounding_boxes, segmentations_df, video_output_path)

                if out is not None or inference_videos_config['show_labeled_frame']:
                    frame = \
                        draw_bounding_boxes_on_image(frame, detected_bounding_boxes + segmentation_bounding_boxes)

                    if out is not None:
                        out.write(frame)

                    if inference_videos_config['show_labeled_frame']:
                        cv2.imshow(video_name, frame)
                        if cv2.waitKey(1) == ord('q'):
                            break

                print(f'Frame {frame_id} processed.')
                frame_id += 1

            cap.release()
            if out is not None:
                out.release()

        elif mode_selected_option == MODE_LABEL_IMAGES:
            assert INFERENCE_DETECTION in inference_selected_options \
                and INFERENCE_CLASSIFICATION in inference_selected_options, \
                'Detection and Classification have to be included as inference option in config to label images.'

            df = get_gtsrb_df() if image_labeling_config['create_new_df'] \
                else pd.read_csv(image_labeling_config['save_path'])
            paths = df['Path'].tolist()

            for image_name in os.listdir(image_labeling_config['images_folder_path']):
                if '{}/{}'.format(image_labeling_config['image_prefix_path'], image_name) in paths:
                    continue

                image_path = '{}/{}'.format(image_labeling_config['images_folder_path'], image_name)

                image = load_and_transform_image(image_path, None)
                detected_bounding_boxes, _, _, _ = inference_dispatcher.dispatch(image)
                df = label_image(image_labeling_config, image, image_path, df, detected_bounding_boxes,
                                 config['detection']['first_object_class_id'])

        cv2.waitKey(0)

    elif mode_selected_option == MODE_TRAIN:
        train_options = [(TRAIN_DETECTION, 'Detection'), (TRAIN_CLASSIFICATION, 'Classification')]
        train_selected_option = input_from_options('Train function', train_options,
                                                   startup_config['function_selected_options'])

        if train_selected_option == TRAIN_DETECTION:
            _ = RoadSignDetection(config['detection'], mode='train')

        elif train_selected_option == TRAIN_CLASSIFICATION:
            _ = RoadSignClassification(config['classification'], mode='train')

    elif mode_selected_option == MODE_PLAY_LABELED_VIDEOS:
        videos_path = inference_videos_config['path']
        videos_output_path = inference_videos_config['output_path']
        detection_label_names = read_file_lines(config['classification']['label_names_path'])
        segmentation_label_names = read_file_lines(config['segmentation']['label_names_path'])
        segmentation_label_colors = read_file_lines(config['segmentation']['colors_path'])

        for video_name in os.listdir(videos_path):
            video_path = '{}/{}'.format(videos_path, video_name)
            video_output_path = '{}/{}'.format(videos_output_path, video_name)
            detections_df, segmentations_df = get_video_dfs(video_output_path)
            cap = cv2.VideoCapture(video_path)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            for frame_id in range(frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                _, frame = cap.read()

                detected_bounding_boxes = get_video_df_frame_bounding_boxes(
                    detections_df, frame_id, detection_label_names)
                segmentation_bounding_boxes = get_video_df_frame_bounding_boxes(
                    segmentations_df, frame_id, segmentation_label_names, segmentation_label_colors)
                frame = draw_bounding_boxes_on_image(frame, detected_bounding_boxes + segmentation_bounding_boxes)

                cv2.imshow(video_name, frame)
                cv2.waitKey(1)

            cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    config = load_configuration()

    seed = config['startup']['seed']
    ia.seed(seed)
    np.random.seed(seed)

    display_menu(config)
