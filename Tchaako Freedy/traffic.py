import cv2
import numpy as np
import os
import sys

import tensorflow as tf

# Verify TensorFlow installation
print("TensorFlow version:", tf.__version__)

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [best_model.h5]")

    # Verify data directory exists and is absolute path
    data_dir = os.path.abspath(sys.argv[1])
    if not os.path.isdir(data_dir):
        sys.exit(f"Error: {data_dir} is not a directory")
    
    print(f"Using data directory: {data_dir}")
    
    # Get image arrays and labels for all image files
    try:
        images, labels = load_data(data_dir)
    except ValueError as e:
        sys.exit(str(e))

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    """
    images = []
    labels = []
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.ppm')
    
    print(f"Loading data from {data_dir}")
    print("Supported formats:", supported_formats)
    total_images = 0
    categories_found = 0
    
    # Check if directory exists and list contents
    if not os.path.exists(data_dir):
        raise ValueError(f"Directory {data_dir} does not exist")
    
    contents = os.listdir(data_dir)
    print(f"Found {len(contents)} items in directory")
    
    # Iterate through each category folder (0 to NUM_CATEGORIES-1)
    for label in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(label))
        
        # Make sure category directory exists
        if not os.path.exists(category_dir):
            print(f"Warning: Category directory {label} does not exist")
            # Try to find actual directory structure
            if label == 0:
                print("Searching for alternative directory structure...")
                for item in contents:
                    full_path = os.path.join(data_dir, item)
                    if os.path.isdir(full_path):
                        print(f"Found directory: {item}")
            continue
        
        # Count files in category
        try:
            files = [f for f in os.listdir(category_dir) 
                    if f.lower().endswith(supported_formats)]
        except Exception as e:
            print(f"Error reading directory {category_dir}: {str(e)}")
            continue
            
        if files:
            categories_found += 1
            print(f"Category {label}: Found {len(files)} image files")
        
        # Iterate through images in the category directory
        for filename in files:
            file_path = os.path.join(category_dir, filename)
            
            try:
                # Read image using OpenCV
                image = cv2.imread(file_path)
                
                if image is None:
                    print(f"Warning: Could not read image {file_path}")
                    # Try reading PPM files differently if needed
                    if filename.lower().endswith('.ppm'):
                        try:
                            image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 
                                               cv2.IMREAD_COLOR)
                        except Exception as e:
                            print(f"PPM reading failed: {str(e)}")
                            continue
                
                if image is None:
                    continue
                    
                # Resize image to the specified dimensions
                image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                
                # Append resized image and its label
                images.append(image_resized)
                labels.append(label)
                total_images += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
    
    if total_images == 0:
        raise ValueError(f"""No valid images found in any category.
        Directory searched: {data_dir}
        Categories found: {categories_found}
        Supported formats: {supported_formats}""")
    
    print(f"Successfully loaded {total_images} images from {categories_found} categories")
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Normalize pixel values to be between 0 and 1
    images = images.astype('float32') / 255.0
    
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model with batch normalization
    and a deeper architecture.
    """
    model = tf.keras.Sequential([
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])
    
    # Compile with a better optimizer configuration
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


if __name__ == "__main__":
    main()
