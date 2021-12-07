import pandas as pd
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split

from utils import *

np.random.seed(1)


class RoadSignClassification:
    def __init__(self, test_data_path='data/classification/test'):
        self.labels = read_file_lines('labels.txt')
        self.label_names = read_file_lines('label_names.txt')

        self.shape = (100, 100)
        self.load_saved_model = True
        self.load_model_path = 'data/classification/models/my_model'
        self.save_trained_model = False
        self.save_trained_model_path = 'data/classification/models/new_model'

        if test_data_path is not None:
            # Obtaining testing data
            print("Obtaining testing data...")
            self.test_images, self.test_labels = self.get_data(test_data_path)

        # Obtaining model
        if self.load_saved_model:
            print("Loading model...")
            self.model = self.load_model(self.load_model_path)
        else:
            self.model = self.get_model()

            # Obtaining training data
            print("Obtaining training data...")
            self.train_images, self.train_labels = self.get_data('data/classification/train')
            # Turn a single categorical column into many indicator columns (A-1, A-11, A-11a, ...)
            self.train_labels = pd.get_dummies(self.train_labels).values

            # Splitting training data into train and validation datasets
            print("Splitting training data...")
            train_x, train_y, validation_x, validation_y = train_test_split(
                self.train_images, self.train_labels, random_state=1)

            print("Training model...")
            self.train_model(train_x, train_y, validation_x, validation_y)

            evaluate = self.evaluate_model(validation_x, validation_y)
            print(evaluate)

            self.show_history_metric('acc')
            self.show_history_metric('loss')

            if self.save_trained_model:
                print("Saving model...")
                self.save_model()

        self.model.summary()
        # self.show_history()

    def load_and_resize_image(self, path):
        return load_and_transform_image(path, self.shape)

    def get_data(self, path):
        images = []
        image_labels = []

        for sign_code in os.listdir(path):
            for sign_image_name in os.listdir('{}/{}'.format(path, sign_code)):
                sign_image_path = '{}/{}'.format(sign_code, sign_image_name)
                image = self.load_and_resize_image(os.path.join(path, sign_image_path))
                images.append(image)
                image_labels.append(sign_code)

        return np.array(images), image_labels

    def get_model(self):
        model = Sequential()
        model.add(Conv2D(kernel_size=(3, 3), filters=32, activation='tanh', input_shape=(*self.shape, 3,)))
        model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))
        model.add(MaxPool2D(2, 2))
        model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))
        model.add(MaxPool2D(2, 2))
        model.add(Conv2D(filters=30, kernel_size=(3, 3), activation='tanh'))

        model.add(Flatten())

        model.add(Dense(20, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(len(self.labels), activation='softmax'))

        model.compile(
            loss='categorical_crossentropy',
            metrics=['acc'],
            optimizer='adam'
        )

        return model

    def train_model(self, train_x, train_y, validation_x, validation_y, epochs=50, batch_size=50):
        csv_logger = CSVLogger('{}.log'.format(self.save_trained_model_path), separator=',', append=False)

        return self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                              validation_data=(validation_x, validation_y), callbacks=[csv_logger])

    def save_model(self):
        self.model.save(self.save_trained_model_path)

    def load_model(self, path):
        return tensorflow.keras.models.load_model(path)

    def show_history(self):
        if self.load_saved_model:
            print(pd.read_csv('{}.log'.format(self.load_model_path), sep=',', engine='python'))
        else:
            print(self.model.history)

    def show_history_metric(self, metric):
        plt.plot(self.model.history[metric])
        plt.plot(self.model.history['val_{}'.format(metric)])
        plt.title('model {}'.format(metric))
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def evaluate_model(self, validation_x, validation_y):
        # The loss value & metrics values for the model.
        return self.model.evaluate(validation_x, validation_y)

    def model_predict_test_data(self, show_images=False):
        predicted = 0
        total = 0

        for index in range(len(self.test_images)):
            image = self.test_images[index:index+1]
            image_label = self.test_labels[index:index+1][0]
            max_label_index = self.model_predict_max_label_index(image)
            predicted_label = self.labels[max_label_index]
            predicted_label_name = self.label_names[max_label_index]

            total = total + 1

            if predicted_label == image_label:
                predicted = predicted + 1

            if show_images:
                title = '{} ({})'.format(predicted_label_name, predicted_label == image_label)
                show_image_with_title(image, title)

        print('Prediction accuracy: {}% ({}/{}).'.format(predicted*100/total, predicted, total))

    def model_predict_data(self, images, show_images=False):
        predicted_labels = []

        for image in images:
            image = cv2.resize(image, self.shape)
            max_label_index = self.model_predict_max_label_index(image)
            predicted_label_name = self.label_names[max_label_index]
            predicted_labels.append(predicted_label_name)

            if show_images:
                show_image_with_title(image, predicted_label_name)

        return predicted_labels

    def model_predict_max_label_index(self, image):
        if len(image.shape) == 3:
            image = tensorflow.expand_dims(image, axis=0)

        predict = self.model.predict(np.array(image))

        # Find the class label that has the greatest probability for each image pixel
        return np.argmax(predict)
