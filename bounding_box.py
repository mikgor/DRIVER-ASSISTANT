import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np


class BoundingBox:
    def __init__(self, start_x, start_y, end_x, end_y, label_id=None,
                 label_name=None, label_color=(0, 0, 255), score=None, image=None):
        self.start_x = int(start_x)
        self.start_y = int(start_y)
        self.end_x = int(end_x)
        self.end_y = int(end_y)
        self.label_id = label_id
        self.label_name = label_name
        self.label_color = tuple((int(x) for x in label_color.split(','))) if type(label_color) is str else label_color
        self.score = score
        self.image = image

    def get_coords(self):
        return self.start_x, self.start_y, self.end_x, self.end_y

    def get_as_df_row(self):
        return [*self.get_coords(), self.label_id, self.score]

    def draw_on_image(self, image, with_label_id=False, thickness=2, margin=5):
        (x, y, w, h) = (self.start_x, self.start_y, self.end_x - self.start_x, self.end_y - self.start_y)
        text = self.label_name.capitalize() if self.label_name else ''

        if with_label_id:
            text += f'({self.label_id})'

        font_path = "arial.ttf"
        font = ImageFont.truetype(font_path, 16)
        text_width, text_height = font.getsize(text)

        org_x = x - text_width / 2 + w / 2
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

        color_bgr = self.label_color if self.label_color is not None else (0, 0, 255)
        shadow_color = "#000"

        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        # Shadow
        draw.text((org[0] + 1, org[1] + 1), text, font=font, fill=shadow_color)
        draw.text((org[0] - 1, org[1] - 1), text, font=font, fill=shadow_color)
        draw.text((org[0] - 1, org[1] + 1), text, font=font, fill=shadow_color)
        draw.text((org[0] + 1, org[1] - 1), text, font=font, fill=shadow_color)

        draw.text(org, text, font=font, fill="#%02x%02x%02x" % color_bgr)

        img_pil = cv2.rectangle(np.array(img_pil), (x, y), (x + w, y + h), color_bgr, thickness)

        return np.array(img_pil)
