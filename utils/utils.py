import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import tensorflow
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
import pandas as pd
import albumentations
from albumentations.pytorch import ToTensorV2

from detection.bounding_box import BoundingBox


def read_file_lines(path):
    return open(path, encoding="utf-8").read().splitlines()


def load_and_transform_image(path, shape):
    image = cv2.imread(path)

    if shape is not None:
        image = cv2.resize(image, shape)

    return image


def image_to_rgb(image):
    # OpenCV reads images into Numpy arrays and stores the channels in BGR order
    # Matplotlib stores in RGB
    return image[..., ::-1]


def show_image_with_title(image, title):
    if len(image.shape) == 4:
        image = tensorflow.squeeze(image)

    plt.imshow(image_to_rgb(image))
    plt.title(title)
    plt.show()


def create_directories_for_labels(path, labels):
    for label in labels:
        os.mkdir('{}/{}'.format(path, label))


def create_bounding_boxes(coordinates, label_ids=None, label_names=None, label_colors=None, scores=None, images=None):
    if label_ids is None:
        label_ids = []
    if label_names is None:
        label_names = []
    if label_colors is None:
        label_colors = []
    if scores is None:
        scores = []
    if images is None:
        images = []

    coordinates_len = len(coordinates)
    label_ids_len = len(label_ids)
    label_names_len = len(label_names)
    label_colors_len = len(label_colors)
    scores_len = len(scores)
    images_len = len(images)

    bounding_boxes = []
    for (index, (start_x, start_y, end_x, end_y)) in enumerate(coordinates):
        bounding_box = BoundingBox(start_x, start_y, end_x, end_y,
                                   label_ids[index] if label_ids_len == coordinates_len else None,
                                   label_names[index] if label_names_len == coordinates_len else None,
                                   label_colors[index] if label_colors_len == coordinates_len else None,
                                   scores[index] if scores_len == coordinates_len else None,
                                   images[index] if images_len == coordinates_len else None)
        bounding_boxes.append(bounding_box)

    return bounding_boxes


def draw_bounding_boxes_on_image(image, bounding_boxes, with_label_id=False, thickness=2, margin=5):
    img = image.copy()
    for bounding_box in bounding_boxes:
        img = bounding_box.draw_on_image(img, with_label_id, thickness, margin)

    return img


def draw_masks_on_image(image, masks):
    img = image.copy()
    for mask in masks:
        img = mask.draw_mask_on_image(img)

    return img


def get_gtsrb_df():
    return pd.DataFrame(columns=['Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId', 'Path'])


def read_gtsrb_csv_row(row):
    width = row['Width']
    height = row['Height']
    start_x = row['Roi.X1']
    start_y = row['Roi.Y1']
    end_x = row['Roi.X2']
    end_y = row['Roi.Y2']
    class_id = row['ClassId']
    path = row['Path']

    return width, height, path, BoundingBox(start_x, start_y, end_x, end_y, label_id=class_id)


def add_prefix_before_file_extension(file_path, prefix):
    file_path_dot_split = file_path.split('.')
    file_path_dot_split.insert(-1, str(prefix))

    return '.'.join(file_path_dot_split)


def add_timestamp_before_file_extension(file_path):
    return add_prefix_before_file_extension(file_path, time.time())


def combine_images_horizontally(image1, image2, image2_bounding_boxes=None, margin=None):
    if margin is None:
        margin = random.randint(0, 200)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    combined = np.zeros((max(h1, h2), w1 + w2 + margin, 3), np.uint8)

    combined[:h1, :w1, :3] = image1
    combined[:h2, w1 + margin:w1 + w2 + margin, :3] = image2

    updated_bounding_boxes = []
    if image2_bounding_boxes is not None:
        for bounding_box in image2_bounding_boxes:
            (start_x, start_y, end_x, end_y) = bounding_box
            start_x = start_x + w1 + margin
            end_x = end_x + w1 + margin
            updated_bounding_boxes.append((start_x, start_y, end_x, end_y))

    return combined, updated_bounding_boxes


def augment_dataset(augment_type, images, bounding_boxes=None, class_ids=None, image_paths=None):
    def augment_duplicate_contrast_and_combine(n_images, n_bounding_boxes=None,
                                               n_class_ids=None, n_image_paths=None):
        images_copy = n_images.copy()
        bounding_boxes_copy = None if n_bounding_boxes is None else n_bounding_boxes.copy()

        seq = iaa.Sequential([
            iaa.LinearContrast((0.75, 1.5)),
            iaa.BlendAlpha(0.5, iaa.Grayscale(1.0)),
            iaa.GammaContrast((0.5, 2.0)),
            iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
        ], random_order=True)

        images_aug, bounding_boxes_aug = seq(images=images_copy, bounding_boxes=bounding_boxes_copy)
        n_images += images_aug

        if n_bounding_boxes is not None:
            n_bounding_boxes += bounding_boxes_aug

        if n_class_ids is not None:
            n_class_ids += n_class_ids

        if n_image_paths is not None:
            n_image_paths += n_image_paths

        return n_images, n_bounding_boxes, n_class_ids, n_image_paths

    def augment_resize_and_pad(n_images, n_bounding_boxes=None):
        seq = iaa.Sequential([
            iaa.Resize((0.9, 1.5)),
            iaa.PadToFixedSize(width=600, height=400, pad_mode=[
                "constant", "edge", "maximum", "mean", "median", "minimum"]),
        ])

        return seq(images=n_images, bounding_boxes=n_bounding_boxes)

    if augment_type == 0:
        return augment_duplicate_contrast_and_combine(images, bounding_boxes, class_ids, image_paths)
    elif augment_type == 1:
        return *augment_resize_and_pad(images, bounding_boxes), class_ids, image_paths

    return images, bounding_boxes, class_ids, image_paths


def generate_augmented_images_and_bounding_boxes_dataset(data_annotations_path, data_path='data/detection/',
                                                         augmented_data_annotations_path_infix='_augmented',
                                                         augmented_data_path='Augmented/', augment_type=0,
                                                         combine_randomly=False, combined_class_id='x'):
    data = pd.read_csv(data_annotations_path)
    images = []
    image_paths = []
    class_ids = []
    bounding_boxes = []

    for index, row in data.iterrows():
        _, _, path, bounding_box = read_gtsrb_csv_row(row)
        c_ids = [bounding_box.label_id]
        boxes = [ia.BoundingBox(*bounding_box.get_coords())]

        image = cv2.imread(data_path + path)

        if combine_randomly:
            number_of_combined_images = random.randint(1, 3)
            if number_of_combined_images > 1:
                path_split = path.split('/')

                if path_split[-2].isnumeric():  # path contains class id
                    path = '/'.join(path_split[0:-2]) + '/{}/'.format(combined_class_id) + path_split[-1]
                else:
                    path = '/'.join(path_split[0:-1]) + '/' + path_split[-1]

                for n in range(number_of_combined_images-1):
                    random_image_index = random.randint(0, len(data)-1)
                    random_row = data.loc[random_image_index]
                    _, _, random_path, random_bounding_box = read_gtsrb_csv_row(random_row)

                    c_ids.append(random_bounding_box.label_id)
                    random_img_path = data_path + random_path
                    random_image = cv2.imread(random_img_path)
                    path = path + random_img_path.split('/')[-1]

                    random_image_coords = (random_bounding_box.get_coords())
                    image, random_image_coords =\
                        combine_images_horizontally(image, random_image, [random_image_coords])

                    for coords in random_image_coords:
                        (start_x, start_y, end_x, end_y) = coords
                        boxes.append(ia.BoundingBox(x1=start_x, y1=start_y, x2=end_x, y2=end_y))

        images.append(image)
        image_paths.append(path)
        class_ids.append(c_ids)
        bounding_boxes.append(boxes)

    images_aug, bbs_aug, class_ids, image_paths = \
        augment_dataset(augment_type, images, bounding_boxes, class_ids, image_paths)

    df = get_gtsrb_df()

    for (index, image) in enumerate(images_aug):
        for (o, bbox) in enumerate(bbs_aug[index]):
            start_x = int(bbox.x1)
            start_y = int(bbox.y1)
            end_x = int(bbox.x2)
            end_y = int(bbox.y2)
            img_name = add_timestamp_before_file_extension(image_paths[index])
            img_path = os.path.join(data_path, augmented_data_path, img_name)
            height, width = image.shape[:2]

            os.makedirs('/'.join(img_path.split('/')[0:-1]), exist_ok=True)
            cv2.imwrite(img_path, image)

            df.loc[len(df)] = [width, height, start_x, start_y, end_x, end_y, class_ids[index][o],
                               os.path.join(augmented_data_path, img_name)]

    data_annotations_path_split = data_annotations_path.split('.')
    augmented_data_annotations_path = add_timestamp_before_file_extension(
        data_annotations_path.split('.')[0] + augmented_data_annotations_path_infix
        + '.' + data_annotations_path_split[1]
    )

    df.to_csv(augmented_data_annotations_path, index=False)

    return augmented_data_annotations_path


def collate_fn(data):
    return tuple(zip(*data))


def transform_to_tensor_v2():
    return albumentations.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def get_video_detections_df():
    return pd.DataFrame(columns=['FrameId', 'X1', 'Y1', 'X2', 'Y2', 'LabelId', 'Score'])


def get_video_segmentations_df():
    return pd.DataFrame(columns=['FrameId', 'X1', 'Y1', 'X2', 'Y2', 'LabelId'])


def get_video_dfs(video_output_path):
    detections_df_path = video_output_path + '_detections.csv'
    segmentations_df_path = video_output_path + '_segmentations.csv'

    detections_df = \
        pd.read_csv(detections_df_path) if os.path.exists(detections_df_path) else get_video_detections_df()
    segmentations_df = \
        pd.read_csv(segmentations_df_path) if os.path.exists(segmentations_df_path) else get_video_segmentations_df()

    return detections_df, segmentations_df


def save_video_frame_dfs(frame_id, detection_bounding_boxes, detections_df,
                         segmentation_bounding_boxes, segmentations_df, video_output_path):
    for bounding_box in detection_bounding_boxes:
        detections_df.loc[len(detections_df)] = [frame_id] + bounding_box.get_as_df_row()

    for bounding_box in segmentation_bounding_boxes:
        segmentations_df.loc[len(segmentations_df)] = [frame_id] + bounding_box.get_as_df_row()[:5]

    detections_df.to_csv(video_output_path + '_detections.csv', index=False)
    segmentations_df.to_csv(video_output_path + '_segmentations.csv', index=False)


def get_video_df_frame_bounding_boxes(video_df, frame_id, label_names, label_colors=None):
    bounding_boxes = []
    frame_rows = video_df.loc[video_df['FrameId'] == frame_id]

    for _, row in frame_rows.iterrows():
        label_id = int(row['LabelId'])
        bounding_box = BoundingBox(row['X1'], row['Y1'], row['X2'], row['Y2'], row['LabelId'],
                                   label_names[label_id] if label_names else None,
                                   label_colors[label_id] if label_colors else None,
                                   row['Score'] if 'Score' in video_df.columns else None)
        bounding_boxes.append(bounding_box)

    return bounding_boxes


def create_sign_classification_dataset_from_gtsrb_df(df_path, destination_path, detection_config,
                                                     classification_config):
    df = pd.read_csv(df_path)

    labels = read_file_lines(classification_config['labels_path'])

    index = 0
    for _, row in df.iterrows():
        _, _, path, bounding_box = read_gtsrb_csv_row(row)
        image = load_and_transform_image(os.path.join(detection_config['data_path'], path), None)
        box_image = image.astype("uint8")[bounding_box.start_y:bounding_box.end_y,
                                          bounding_box.start_x:bounding_box.end_x]

        image_name = path.split('/')[-1]
        save_dir_path = os.path.join(destination_path,
                                     labels[bounding_box.label_id-detection_config['first_object_class_id']])
        save_path = add_prefix_before_file_extension(os.path.join(save_dir_path, image_name), index)
        os.makedirs(save_dir_path, exist_ok=True)
        cv2.imwrite(save_path, box_image)
        index = index + 1

        print(save_path)
