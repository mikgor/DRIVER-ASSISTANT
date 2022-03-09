import os

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.auto import tqdm
import torch
import matplotlib.pyplot as plt
import time

from detection.bounding_box import BoundingBox
from utils.utils import collate_fn, transform_to_tensor_v2, read_file_lines
from utils.utils import read_gtsrb_csv_row, generate_augmented_images_and_bounding_boxes_dataset


class RoadSignDataset(Dataset):
    def __init__(self, data_path, annotations_path, shape, classes):
        self.height, self.width = shape
        self.classes = classes
        self.all_images = []

        # More classes than Background and Object
        self.multi_labels = len(classes) > 2

        annotations = pd.read_csv(annotations_path)

        skipped_indices = []
        for index, row in annotations.iterrows():
            if index in skipped_indices:
                skipped_indices.remove(index)
            else:
                bounding_boxes = []
                path = row['Path']

                path_all_bounding_boxes_rows = annotations.loc[annotations['Path'] == path]
                skipped_indices = skipped_indices + list(path_all_bounding_boxes_rows.index.values)[1:]

                for _, bounding_box_row in path_all_bounding_boxes_rows.iterrows():
                    _, _, _, bounding_box = read_gtsrb_csv_row(bounding_box_row)
                    if not self.multi_labels:
                        bounding_box.label_id = 1
                    bounding_boxes.append(bounding_box)

                img_path = data_path + path

                self.all_images.append((img_path, bounding_boxes))

    def __getitem__(self, idx):
        img_path, bounding_boxes = self.all_images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        image_height = image.shape[0]
        image_width = image.shape[1]

        boxes = []
        labels = []

        for box in bounding_boxes:
            start_x = (box.start_x / image_width) * self.width
            end_x = (box.end_x / image_width) * self.width
            start_y = (box.start_y / image_height) * self.height
            end_y = (box.end_y / image_height) * self.height
            boxes.append([start_x, start_y, end_x, end_y])
            labels.append(box.label_id)

        # Boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes_area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        is_crowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        target = {"boxes": boxes, "labels": labels, "area": boxes_area, "iscrowd": is_crowd, "image_id": image_id}

        sample = transform_to_tensor_v2()(image=image_resized, bboxes=target['boxes'], labels=labels)
        image_resized = sample['image']
        target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target

    def __len__(self):
        return len(self.all_images)


class RoadSignDetection:
    def __init__(self, config, mode='train'):

        self.batch_size = config['batch_size']
        self.shape = (config['shape'], config['shape'])
        self.epochs = config['epochs']
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.classes = ['background'] + [str(c) for c in range(config['first_object_class_id'],
                                         len(read_file_lines(config['labels_path'])) + config['first_object_class_id'])]
        self.classes_number = len(self.classes)
        self.model_path = config['model_path']
        self.model_dir_path = config['model_dir_path']
        self.save_plot_after_x_epochs = config['save_plot_after_x_epochs']
        self.save_model_after_x_epochs = config['save_model_after_x_epochs']

        if mode == 'train':
            self.train_data_loader = self.get_data_loader('Training', config['data_path'],
                                                          config['train_data_annotations_path'],
                                                          config['datasets_augmentation'], shuffle=True)

            self.validation_data_loader = self.get_data_loader('Validating', config['data_path'],
                                                               config['validation_data_annotations_path'],
                                                               config['datasets_augmentation'])

            self.train_model()

        elif mode == 'inference':
            self.model = self.load_model(self.model_path)

    def get_data_loader(self, name, data_path, data_annotations_path, config, shuffle=False):
        print(f"Obtaining {name} data...")

        if config['augment_datasets']:
            print(f"Obtaining augmented {name} data...")
            data_annotations_path = generate_augmented_images_and_bounding_boxes_dataset(
                data_annotations_path, data_path, combine_randomly=config['combine_randomly'])

        dataset = RoadSignDataset(data_path, data_annotations_path, self.shape, self.classes)
        print(f"{name} data length: {len(dataset)}")

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0, collate_fn=collate_fn)

    def __epoch_train(self, model, train_iterations, train_loss_list, train_loss_total, train_loss_iterations, optimizer):
        print('Training...')

        progress_bar = tqdm(self.train_data_loader, total=len(self.train_data_loader))

        for i, data in enumerate(progress_bar):
            optimizer.zero_grad()
            images, targets = data

            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss_list.append(loss_value)
            train_loss_total += loss_value
            train_loss_iterations += 1
            losses.backward()
            optimizer.step()
            train_iterations += 1

            progress_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        return train_iterations, train_loss_list, train_loss_total, train_loss_iterations, optimizer

    def __epoch_validate(self, model, val_iterations, val_loss_list, validation_loss_total, validation_loss_iterations):
        print('Validating...')

        progress_bar = tqdm(self.validation_data_loader, total=len(self.validation_data_loader))

        for i, data in enumerate(progress_bar):
            images, targets = data

            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            val_loss_list.append(loss_value)
            validation_loss_total += loss_value
            validation_loss_iterations += 1
            val_iterations += 1

            progress_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        return val_iterations, val_loss_list, validation_loss_total, validation_loss_iterations

    def get_model(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.classes_number)
        model.train()
        model = model.to(self.device)

        return model

    def load_model(self, model_path):
        model = self.get_model()
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        return model

    def model_save(self, model, epoch):
        os.makedirs(self.model_dir_path, exist_ok=True)
        torch.save(model.state_dict(), f"{self.model_dir_path}/model{epoch}.pth")
        print('Model saved\n')

    def generate_and_save_loss_plot(self, name, loss_list, epoch, color='blue'):
        figure, ax = plt.subplots()
        ax.plot(loss_list, color=color)
        ax.set_xlabel('iterations')
        ax.set_ylabel(f'{name} loss')
        os.makedirs(self.model_dir_path, exist_ok=True)
        figure.savefig(f"{self.model_dir_path}/{name}_loss_{epoch}.png")

    def generate_and_save_loss_plots(self, train_loss_list, validation_loss_list, epoch):
        plt.style.use('ggplot')

        self.generate_and_save_loss_plot('train', train_loss_list, epoch)
        self.generate_and_save_loss_plot('validation', validation_loss_list, epoch, color='red')

        plt.close('all')
        print('Plots saved\n')

    def train_model(self):
        model = self.get_model()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

        train_loss_total = 0.0
        train_loss_iterations = 0.0
        validation_loss_total = 0.0
        validation_loss_iterations = 0.0
        train_iterations = 1
        validation_iterations = 1
        train_loss_list = []
        val_loss_list = []

        for epoch in range(self.epochs):
            print(f"\nEPOCH {epoch + 1} of {self.epochs}")

            train_loss_total = 0.0
            train_loss_iterations = 0.0
            validation_loss_total = 0.0
            validation_loss_iterations = 0.0

            start = time.time()
            train_iterations, train_loss_list, train_loss_total, train_loss_iterations, optimizer = self.__epoch_train(
                    model, train_iterations, train_loss_list, train_loss_total, train_loss_iterations, optimizer)

            validation_iterations, val_loss_list, validation_loss_total, validation_loss_iterations = \
                self.__epoch_validate(
                    model, validation_iterations, val_loss_list, validation_loss_total, validation_loss_iterations)

            print(f"Epoch #{epoch} train loss: "
                  f"{0 if train_loss_iterations == 0 else 1.0 * train_loss_total / train_loss_iterations:.3f}")
            print(f"Epoch #{epoch} validation loss: "
                  f"{0 if validation_loss_iterations == 0 else 1.0 * validation_loss_total / validation_loss_iterations:.3f}")

            end = time.time()
            print(f"Epoch {epoch} took {((end - start) / 60):.3f} minutes")

            current_epoch = epoch + 1
            if current_epoch == self.epochs or current_epoch % self.save_model_after_x_epochs == 0:
                self.model_save(model, current_epoch)

            if current_epoch == self.epochs or current_epoch % self.save_plot_after_x_epochs == 0:
                self.generate_and_save_loss_plots(train_loss_list, val_loss_list, current_epoch)

    def predict_bounding_boxes(self, image_path, detection_threshold=0.77):
        image = image_path

        if type(image_path) == str:
            image = cv2.imread(image_path)

        original_image = image.copy()
        # Image from BGR to RGB and the pixel range between 0 and 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # Change channel position to first and convert to tensor
        image = np.transpose(image, (2, 0, 1)).astype(np.float)
        image = torch.tensor(image, dtype=torch.float)

        # Add batch dimension
        image = torch.unsqueeze(image, 0)

        bounding_boxes: [BoundingBox] = []

        with torch.no_grad():
            outputs = self.model(image)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        for index in range(len(outputs)):
            for idx, score in enumerate(outputs[index]['scores']):
                if score > detection_threshold:
                    (start_x, start_y, end_x, end_y) = outputs[index]['boxes'][idx].data.numpy().astype(np.int32)
                    bounding_boxes.append(
                        BoundingBox(start_x, start_y, end_x, end_y,
                                    outputs[index]['labels'][idx].data.numpy(),
                                    score=outputs[index]['scores'][idx].data.numpy(),
                                    image=original_image.astype("uint8")[start_y:end_y, start_x:end_x])
                    )

        return bounding_boxes
