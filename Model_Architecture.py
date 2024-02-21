import os
import numpy as np
import tensorflow as tf
import matplotlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image

class Data_Preprocessing():
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.dataset = tf.keras.utils.image_dataset_from_directory(
            directory_path,
            labels='inferred',
            color_mode='grayscale',
            image_size=(492, 400),
            shuffle=True,
            seed=12345,
            batch_size=16,
        )
        self.normalize()
        self.train_data, self.validation_data, self.test_data = self.data_split()

    def normalize(self):
        # Calculate the mean and standard deviation
        pixel_values = []
        for images, _ in self.dataset:
            mean_image = tf.reduce_mean(images, axis=(0, 1, 2))
            pixel_values.append(mean_image.numpy())
        self.mean = np.mean(pixel_values)
        self.standard_deviation = np.std(pixel_values)

        # Normalize the training, validation, and test data
        self.dataset = self.dataset.map(lambda x, y: (x - self.mean) / self.standard_deviation)

    def data_split(self):
        # Calculating the sizes of the training, validation, and test sets
        self.dataset = self.dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset_size = len(list(self.dataset))
        train_size = int(0.8 * dataset_size)
        validation_size = int(0.15 * dataset_size)
        test_size = len(self.dataset) - train_size - validation_size

        # Splitting the dataset using TensorFlow methods "take()" and "split()"
        train_data, validation_test_data = (
            self.dataset.take(train_size),
            self.dataset.skip(train_size),
        )
        validation_data, test_data = (
            validation_test_data.take(validation_size),
            validation_test_data.skip(validation_size),
        )

        return train_data, validation_data, test_data

    
class Model_Creation():
    def __init__(self, train_data, validation_data, test_data):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data

    def setup_model(self):
        # Setup of the Sequential Model
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(492, 400, 1)),
            MaxPooling2D(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(150, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.compile_model()
    
    def visualize_clusters(self):
        predicted_classes = self.model.predict_classes(self.test_data)
        reducer = umap.UMAP(n_components=2)
        embedding = reducer.fit_transform(self.test_data)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=predicted_classes.reshape(-1, 1), cmap='viridis')
        plt.show()

    def compile_model(self):
        # Setup of the compiling
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # Training
        self.model.fit(self.train_data, validation_data=self.validation_data, epochs=9)
        # Saving
        self.model.save(os.path.join("Trained_Models", "Xray_Diagnosis_AI_1.h5"))


    



