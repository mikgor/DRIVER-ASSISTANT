import cv2
import mouse

from detection.bounding_box import BoundingBox
from utils.utils import draw_bounding_boxes_on_image


def mouse_on_click(mouse_click_positions, image_shape, bounding_boxes,
                   window_x, window_y, offset_x, offset_y, default_class_id):
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

            print(f"{mouse_click_positions[-2]}, ({x}, {y}) - {bounding_boxes[-1].label_id}. "
                  f"Press any key to update.")


def mouse_on_right_click(image_shape, bounding_boxes, window_x, window_y, offset_x, offset_y):
    h, w, _ = image_shape
    x, y = mouse.get_position()
    x -= offset_x
    y -= offset_y

    if w + window_x >= x >= window_x and h + window_y >= y >= window_y:
        x -= window_x
        y -= window_y

        for bounding_box in bounding_boxes:
            start_x, start_y, end_x, end_y = bounding_box.get_coords()

            if start_x <= x <= end_x and start_y <= y <= end_y:
                print(f'Removed box label_id: {bounding_box.label_id}. Press any key to update.')
                bounding_boxes.remove(bounding_box)
                break


def update_and_show_image(image, image_path, bounding_boxes, window_x, window_y):
    image = draw_bounding_boxes_on_image(image, bounding_boxes, True)
    cv2.namedWindow(image_path)
    cv2.moveWindow(image_path, window_x, window_y)
    cv2.imshow(image_path, image)


def label_image(config, image, image_path, df, bounding_boxes, first_object_class_id=0):
    image_name = image_path.split('/')[-1]
    shape = image.shape
    mouse_click_positions = []
    window_x = config['window_x']
    window_y = config['window_y']
    offset_x = config['offset_x']
    offset_y = config['offset_y']

    mouse.on_click(mouse_on_click, args=(mouse_click_positions, shape, bounding_boxes,
                                         window_x, window_y, offset_x, offset_y, config['default_class_id']))
    mouse.on_right_click(mouse_on_right_click, args=(shape, bounding_boxes, window_x, window_y, offset_x, offset_y))

    update_and_show_image(image, image_path, bounding_boxes, window_x, window_y)

    while True:
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key == 122:  # z
            if len(bounding_boxes) > 0:
                print(f'Removed box label_id: {bounding_boxes[-1].label_id}.')
                bounding_boxes.pop()
                update_and_show_image(image, image_path, bounding_boxes, window_x, window_y)
        elif key == 32:  # space
            for bounding_box in bounding_boxes:
                df.loc[len(df)] = [shape[1], shape[0], *bounding_box.get_coords(),
                                   int(bounding_box.label_id) + first_object_class_id,
                                   '{}/{}'.format(config['image_prefix_path'], image_name)]
            df.to_csv(config['save_path'], index=False)
            break
        else:
            update_and_show_image(image, image_path, bounding_boxes, window_x, window_y)

    mouse.unhook_all()

    return df
