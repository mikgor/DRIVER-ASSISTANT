import os
import cv2
import pandas as pd
import mouse

from bounding_box import BoundingBox
from utils import get_gtsrb_df, load_and_transform_image, draw_bounding_boxes_on_image

images_folder_path = 'data/detection/Train_frames'
image_prefix_path = 'Train_frames'
save_path = 'data/detection/train_frames.csv'
create_new_df = False
window_x = 0
window_y = 230
offset_x = 8
offset_y = 31
default_class_id = 1


def mouse_on_click(mouse_click_positions, image_shape, bounding_boxes):
    h, w, _ = image_shape
    x, y = mouse.get_position()
    x -= offset_x
    y -= offset_y

    if w + window_x >= x >= window_x and h + window_y >= y >= window_y:
        x -= window_x
        y -= window_y
        mouse_click_positions.append((x, y))
        print(x, y)

        if len(mouse_click_positions) % 2 == 0:
            bounding_boxes.append(
                BoundingBox(*mouse_click_positions[-2], *mouse_click_positions[-1], label_id=default_class_id))

            provided_id = input(f"Class ID for {mouse_click_positions[-2]}, ({x}, {y}): ")
            if provided_id != '':
                bounding_boxes[-1].label_id = provided_id

            print(f"{mouse_click_positions[-2]}, ({x}, {y}) - {bounding_boxes[-1].label_id}")


def update_and_show_image(image, image_path, bounding_boxes):
    image = draw_bounding_boxes_on_image(image, bounding_boxes, True)
    cv2.namedWindow(image_path)
    cv2.moveWindow(image_path, window_x, window_y)
    cv2.imshow(image_path, image)


df = get_gtsrb_df() if create_new_df else pd.read_csv(save_path)
paths = df['Path'].tolist()

for image_name in os.listdir(images_folder_path):
    if '{}/{}'.format(image_prefix_path, image_name) in paths:
        continue

    image_path = '{}/{}'.format(images_folder_path, image_name)
    image = load_and_transform_image(image_path, None)
    shape = image.shape
    bounding_boxes = []
    mouse_click_positions = []
    mouse.on_click(mouse_on_click, args=(mouse_click_positions, shape, bounding_boxes))

    update_and_show_image(image, image_path, [])

    while True:
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 122: # z
            if len(bounding_boxes) > 0:
                bounding_boxes.pop()
                update_and_show_image(image, image_path, bounding_boxes)
        elif key == 32: # space
            for bounding_box in bounding_boxes:
                df.loc[len(df)] = [shape[1], shape[0], *bounding_box.get_coords(), bounding_box.label_id,
                                   '{}/{}'.format(image_prefix_path, image_name)]
            df.to_csv(save_path, index=False)
            break
        else:
            update_and_show_image(image, image_path, bounding_boxes)

    mouse.unhook_all()
