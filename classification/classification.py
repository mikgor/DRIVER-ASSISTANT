from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split

from utils.utils import *


class RoadSignClassification:
    def __init__(self, config, mode='train'):
        self.labels = read_file_lines(config['labels_path'])
        self.label_names = read_file_lines(config['label_names_path'])

        self.shape = (config['shape'], config['shape'])
        self.model_path = config['model_path']
        self.save_trained_model = config['save_trained_model']
        self.save_trained_model_path = config['save_trained_model_path']
        self.test_data_paths = config['test_data_paths']

        if mode == 'train':
            self.model = self.get_model()

            # Obtaining training data
            print("Obtaining training data...")
            self.train_images, self.train_labels = self.get_data(
                config['train_data_paths'], config['datasets_augmentation'])

            # Turn a single categorical column into many indicator columns (A-1, A-11, A-11a, ...)
            self.train_labels = pd.get_dummies(self.train_labels).values

            # Splitting training data into train and validation datasets
            print("Splitting training data...")
            train_x, validation_x, train_y, validation_y = train_test_split(
                self.train_images, self.train_labels, random_state=1)

            print("Training model...")
            self.history = self.train_model(train_x, train_y, validation_x, validation_y,
                                            config['epochs'], config['batch_size'])

            if self.save_trained_model:
                print("Saving model...")
                self.save_model()

            evaluate = self.evaluate_model(validation_x, validation_y)
            print(evaluate)

            self.save_history_metric('acc')
            self.save_history_metric('loss')

        elif mode == 'inference':
            print("Loading model...")
            self.model = self.load_model(self.model_path)

        self.model.summary()
        self.show_history()

    def load_and_resize_image(self, path):
        return load_and_transform_image(path, self.shape)

    def get_data(self, paths, config=None):
        images = []
        image_labels = []

        for path in paths:
            for sign_code in os.listdir(path):
                for sign_image_name in os.listdir('{}/{}'.format(path, sign_code)):
                    sign_image_path = '{}/{}'.format(sign_code, sign_image_name)
                    image = self.load_and_resize_image(os.path.join(path, sign_image_path))
                    images.append(image)
                    image_labels.append(sign_code)

        if config and config['augment_train_dataset']:
            print('Augmenting dataset...')
            images, _, _, _ = augment_dataset(config['augment_type'], images)

            image_labels_copy = image_labels.copy()
            image_labels = image_labels + image_labels_copy

        print(f"Dataset length: {len(images)}")

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

    def train_model(self, train_x, train_y, validation_x, validation_y, epochs=150, batch_size=50):
        csv_logger = CSVLogger('{}.log'.format(self.save_trained_model_path), separator=',', append=False)

        return self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                              validation_data=(validation_x, validation_y), callbacks=[csv_logger])

    def save_model(self):
        self.model.save(self.save_trained_model_path)

    def load_model(self, path):
        return tensorflow.keras.models.load_model(path)

    def show_history(self):
        if self.model_path:
            print(pd.read_csv('{}.log'.format(self.model_path), sep=',', engine='python'))
        else:
            print(self.model.history)

    def save_history_metric(self, metric):
        figure, ax = plt.subplots()
        ax.plot(self.history.history[metric])
        ax.plot(self.history.history['val_{}'.format(metric)])
        ax.set_title('model {}'.format(metric))
        ax.set_ylabel(metric)
        ax.set_xlabel('epoch')
        ax.legend(['train', 'validation'], loc='upper left')
        figure.savefig(os.path.join(self.save_trained_model_path, f'_{metric}.png'))

    def evaluate_model(self, validation_x, validation_y):
        # The loss value & metrics values for the model.
        return self.model.evaluate(validation_x, validation_y)

    def model_predict_test_data(self, show_images=False):
        print("Obtaining testing data...")
        test_images, test_labels = self.get_data(self.test_data_paths)

        predicted = 0
        total = len(test_images)

        for index in range(total):
            image = test_images[index:index+1]
            image_label = test_labels[index:index+1][0]
            max_label_index = self.model_predict_max_label_index(image)
            predicted_label = self.labels[max_label_index]
            predicted_label_name = self.label_names[max_label_index]

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
            predicted_labels.append((max_label_index, predicted_label_name))

            if show_images:
                show_image_with_title(image, predicted_label_name)

        return predicted_labels

    def model_predict_max_label_index(self, image):
        if len(image.shape) == 3:
            image = tensorflow.expand_dims(image, axis=0)

        predict = self.model.predict(np.array(image))

        # Find the class label that has the greatest probability for each image pixel
        return np.argmax(predict)
