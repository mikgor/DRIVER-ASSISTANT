import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import tensorflow
import numpy as np
from PIL import ImageFont, ImageDraw, Image

import imgaug as ia
import imgaug.augmenters as iaa
import pandas as pd
import albumentations
from albumentations.pytorch import ToTensorV2


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


def bounding_box_to_bounding_rect(bounding_box):
    (start_x, start_y, end_x, end_y) = bounding_box
    bounding_rect = (int(start_x), int(start_y), int(end_x - start_x), int(end_y - start_y))

    return bounding_rect


def draw_rectangle_on_image(image, bounding_rect):
    (x, y, w, h) = bounding_rect

    # BGR
    color = (0, 0, 255)

    thickness = 2

    return cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)


def draw_rectangle_on_image_from_bounding_box(image, bounding_box):
    bounding_rect = bounding_box_to_bounding_rect(bounding_box)

    return draw_rectangle_on_image(image, bounding_rect)


def draw_rectangles_and_text_on_image_from_bounding_boxes(image, bounding_boxes, texts):
    img = image.copy()
    
    for (index, bounding_box) in enumerate(bounding_boxes):
        bounding_rect = bounding_box_to_bounding_rect(bounding_box)
        img = draw_rectangle_on_image(img, bounding_rect)
        img = draw_text_under_object_on_image(img, bounding_rect, texts[index])

    return img


def draw_text_under_object_on_image(image, object_bounding_rect, text, margin=5):
    (x, y, w, h) = object_bounding_rect
    text = text.capitalize()

    font_path = "arial.ttf"
    font = ImageFont.truetype(font_path, 16)
    text_width, text_height = font.getsize(text)

    org_x = x - text_width/2 + w/2
    org_y = y + h + margin

    if org_x < 0:
        org_x = x

    if org_y < 0:
        org_y = y - margin

    image_height, image_width, _ = image.shape
    if org_x + text_width > image_width:
        org_x = image_width - text_width

    if org_y + text_height > image_height:
        org_y = y + margin

    org = (org_x, org_y)

    # BGR
    color = "#0000ff"
    shadow_color = "#000"

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)

    # Shadow
    draw.text((org[0]+1, org[1]+1), text, font=font, fill=shadow_color)
    draw.text((org[0]-1, org[1]-1), text, font=font, fill=shadow_color)
    draw.text((org[0]-1, org[1]+1), text, font=font, fill=shadow_color)
    draw.text((org[0]+1, org[1]-1), text, font=font, fill=shadow_color)

    draw.text(org, text, font=font, fill=color)

    return np.array(img_pil)


def draw_text_under_object_on_image_from_bounding_box(image, bounding_box, text, margin=5):
    bounding_rect = bounding_box_to_bounding_rect(bounding_box)

    return draw_text_under_object_on_image(image, bounding_rect, text, margin)


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

    return width, height, start_x, start_y, end_x, end_y, class_id, path


def add_timestamp_before_file_extension(file_path):
    file_path_dot_split = file_path.split('.')

    return file_path_dot_split[0] + '_' + str(time.time()) + '.' + file_path_dot_split[1]


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
        _, _, start_x, start_y, end_x, end_y, class_id, path = read_gtsrb_csv_row(row)
        c_ids = [class_id]
        boxes = [ia.BoundingBox(x1=start_x, y1=start_y, x2=end_x, y2=end_y)]

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
                    _, _, random_start_x, random_start_y, random_end_x, random_end_y, random_class_id, random_path = \
                        read_gtsrb_csv_row(random_row)

                    c_ids.append(random_class_id)
                    random_img_path = data_path + random_path
                    random_image = cv2.imread(random_img_path)
                    path = path + random_img_path.split('/')[-1]

                    random_image_bounding_boxes = (random_start_x, random_start_y, random_end_x, random_end_y)
                    image, random_image_bounding_boxes =\
                        combine_images_horizontally(image, random_image, [random_image_bounding_boxes])

                    for bounding_box in random_image_bounding_boxes:
                        (start_x, start_y, end_x, end_y) = bounding_box
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
    return pd.DataFrame(columns=['FrameId', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId', 'Score'])


def get_video_detections(video_path):
    video_detections_csv = video_path + '.csv'

    if os.path.exists(video_detections_csv):
        return pd.read_csv(video_detections_csv)

    return get_video_detections_df()


def save_video_frame_detections(frame_id, bounding_boxes, label_ids, scores, df, video_path):
    video_detections_csv = video_path + '.csv'

    for (index, box) in enumerate(bounding_boxes):
        start_x, start_y, end_x, end_y = box
        df.loc[len(df)] = [frame_id, start_x, start_y, end_x, end_y, label_ids[index], scores[index]]

    df = df.astype(
        {'FrameId': 'int', 'Roi.X1': 'int', 'Roi.Y1': 'int', 'Roi.X2': 'int', 'Roi.Y2': 'int', 'ClassId': 'int'})

    df.to_csv(video_detections_csv, index=False)


def read_video_csv_row(row):
    frame_id = row['FrameId']
    start_x = row['Roi.X1']
    start_y = row['Roi.Y1']
    end_x = row['Roi.X2']
    end_y = row['Roi.Y2']
    class_id = row['ClassId']
    score = row['Score']

    return frame_id, start_x, start_y, end_x, end_y, class_id, score


def play_video_with_labels(cap, video_df, classification_config):
    label_names = read_file_lines(classification_config['label_names_path'])
    skipped_indices = []

    for index, row in video_df.iterrows():
        bounding_boxes = []
        labels = []
        frame_id = row['FrameId']

        frame_all_bounding_boxes_rows = video_df.loc[video_df['FrameId'] == frame_id]
        skipped_indices = skipped_indices + list(frame_all_bounding_boxes_rows.index.values)[1:]

        for _, bounding_box_row in frame_all_bounding_boxes_rows.iterrows():
            _, start_x, start_y, end_x, end_y, class_id, score = read_video_csv_row(bounding_box_row)
            bounding_boxes.append([start_x, start_y, end_x, end_y])
            labels.append(label_names[int(class_id)])

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        _, frame = cap.read()
        frame = draw_rectangles_and_text_on_image_from_bounding_boxes(frame, bounding_boxes, labels)

        cv2.imshow('Video', frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
