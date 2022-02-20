import os
import cv2
import pandas as pd
import mouse

from utils import get_gtsrb_df, load_and_transform_image, draw_rectangles_and_text_on_image_from_bounding_boxes

images_folder_path = 'data/detection/Train_frames'
image_prefix_path = 'Train_frames'
save_path = 'data/detection/train_frames.csv'
create_new_df = False
window_x = 0
window_y = 230
offset_x = 8
offset_y = 31
default_class_id = 1


def mouse_on_click(mouse_click_positions, image_shape, bounding_boxes, class_ids):
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
            bounding_boxes.append((*mouse_click_positions[-2], *mouse_click_positions[-1]))

            class_ids.append(str(default_class_id))
            provided_id = input(f"Class ID for {mouse_click_positions[-2]}, ({x}, {y}): ")
            if provided_id != '':
                class_ids[-1] = provided_id

            print(f"{mouse_click_positions[-2]}, ({x}, {y}) - {class_ids[-1]}")


def update_and_show_image(image, image_path, bounding_boxes, class_ids):
    image = draw_rectangles_and_text_on_image_from_bounding_boxes(image, bounding_boxes, class_ids)
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
    class_ids = []
    mouse_click_positions = []
    mouse.on_click(mouse_on_click, args=(mouse_click_positions, shape, bounding_boxes, class_ids))

    update_and_show_image(image, image_path, [], [])

    while True:
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 122: # z
            if len(bounding_boxes) > 0 and len(class_ids) > 0:
                bounding_boxes.pop()
                class_ids.pop()
                update_and_show_image(image, image_path, bounding_boxes, class_ids)
        elif key == 32: # space
            for (index, box) in enumerate(bounding_boxes):
                start_x, start_y, end_x, end_y = box
                df.loc[len(df)] = [shape[1], shape[0], start_x, start_y, end_x, end_y, class_ids[index],
                                   '{}/{}'.format(image_prefix_path, image_name)]
            df.to_csv(save_path, index=False)
            break
        else:
            update_and_show_image(image, image_path, bounding_boxes, class_ids)

    mouse.unhook_all()
