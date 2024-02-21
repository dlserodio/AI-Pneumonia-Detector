import tensorflow as tf
from Model_Architecture import Data_Preprocessing, Model_Creation

# Set GPU memory growth to True
# Limit GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]  # Adjust the limit as needed
            )
    except RuntimeError as e:
        print(e)


# Set the directory path where your dataset is located
dataset_directory = "C:/Users/lanzi/Desktop/AI Pneumonia Detector/chest_xrays/Dataset"

# Initialize data preprocessing
data_preprocessing = Data_Preprocessing(dataset_directory)

# Get the train, validation, and test data
train_data, validation_data, test_data = data_preprocessing.train_data, data_preprocessing.validation_data, data_preprocessing.test_data

# Initialize model creation
model_creation = Model_Creation(train_data, validation_data, test_data)

# Setup and compile the model
model_creation.setup_model()

# Visualize clusters
model_creation.visualize_clusters()

# Evaluate the model on the test set
evaluation_results = model_creation.model.evaluate(test_data)

# Display evaluation results
print("Test Loss:", evaluation_results[0])
print("Test Accuracy:", evaluation_results[1])
