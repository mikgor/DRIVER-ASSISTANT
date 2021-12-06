import os

import cv2
import matplotlib.pyplot as plt
import tensorflow
import numpy as np
from PIL import ImageFont, ImageDraw, Image


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


def draw_rectangle_on_image(image, bounding_rect):
    (x, y, w, h) = bounding_rect

    # BGR
    color = (0, 0, 255)

    thickness = 2

    return cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)


def draw_text_under_object_on_image(image, object_bounding_rect, text, margin=5):
    (x, y, w, h) = object_bounding_rect
    text = text.capitalize()

    font_path = "arial.ttf"
    font = ImageFont.truetype(font_path, 16)
    text_width = font.getsize(text)[0]

    org_x = x-text_width/2+w/2
    org_y = y + h + margin

    if org_x < 0:
        org_x = x

    if org_y < 0:
        org_y = y - margin

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
