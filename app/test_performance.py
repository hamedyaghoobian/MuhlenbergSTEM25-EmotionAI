import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("Testing Emotion Recognition Model Performance")
print("TensorFlow version:", tf.__version__)

# Set paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_data_dir = os.path.join(base_dir, "testdata")
data_dir = os.path.join(base_dir, "data")
models_dir = os.path.join(base_dir, "models")
model_path = os.path.join(models_dir, "emotion_model.h5")

# Emotion classes (must match model's output classes)
emotions = ["Happy", "Sad", "Surprised", "Neutral"]
emotion_mapping = {
    'happy': 0,
    'sad': 1,
    'surprise': 2,  # Note: directory is 'surprise' but class is 'Surprised'
    'neutral': 3
}

def test_model(model_path, test_data_dir, batch_size=32):
    """Test the model on the test dataset."""
    print(f"\nTesting model: {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Load model
    try:
        print("Loading model...")
        model = load_model(model_path)
        model.summary()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Set up data generator for testing
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Check that test data directory exists
    if not os.path.exists(test_data_dir):
        print(f"Error: Test data directory not found at {test_data_dir}")
        return
    
    # Count test samples
    total_samples = 0
    for emotion in emotions:
        emotion_dir = os.path.join(test_data_dir, emotion.lower())
        if os.path.exists(emotion_dir):
            samples = len([f for f in os.listdir(emotion_dir) if os.path.isfile(os.path.join(emotion_dir, f))])
            print(f"{emotion}: {samples} test samples")
            total_samples += samples
    
    print(f"Total test samples: {total_samples}")
    
    # Create test generator
    try:
        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    except Exception as e:
        print(f"Error creating test generator: {str(e)}")
        return
    
    # Measure performance
    start_time = time.time()
    
    # Evaluate model
    try:
        loss, accuracy = model.evaluate(test_generator)
        print(f"\nTest accuracy: {accuracy:.4f}")
        print(f"Test loss: {loss:.4f}")
    except Exception as e:
        print(f"Error evaluating model: {str(e)}")
        return
    
    # Get predictions
    try:
        test_generator.reset()
        y_pred = model.predict(test_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Get true labels
        y_true = test_generator.classes
        
        # Ensure the mapping between directory names and class indices is correct
        print("\nClass indices in generator:", test_generator.class_indices)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=emotions, 
                    yticklabels=emotions)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save the confusion matrix plot
        cm_path = os.path.join(base_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to: {cm_path}")
        
        # Display classification report
        report = classification_report(y_true, y_pred_classes, target_names=emotions)
        print("\nClassification Report:")
        print(report)
        
        # Calculate per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        for i, emotion in enumerate(emotions):
            print(f"{emotion} accuracy: {per_class_acc[i]:.4f}")
        
        # Measure execution time
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nExecution time: {execution_time:.2f} seconds")
        
        # Calculate prediction time per image
        prediction_time = execution_time / total_samples
        print(f"Average prediction time per image: {prediction_time*1000:.2f} ms")
        
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return

def train_on_testdata(epochs=15, learning_rate=0.0001, use_augmentation=True):
    """Train a model on the larger testdata dataset."""
    print("\n\nTraining new model on testdata dataset...")
    
    # Set up data generators
    if use_augmentation:
        print("Using data augmentation")
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
    else:
        print("Not using data augmentation")
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
    
    # Create generators
    try:
        train_generator = train_datagen.flow_from_directory(
            test_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = train_datagen.flow_from_directory(
            test_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        print(f"Found {train_generator.samples} training images")
        print(f"Found {validation_generator.samples} validation images")
        
        # Create model
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create model
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(len(emotions), activation='softmax')
        ])
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        print(f"\nTraining model with {epochs} epochs, learning rate {learning_rate}...")
        start_time = time.time()
        
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save model
        large_model_path = os.path.join(models_dir, "large_emotion_model.h5")
        model.save(large_model_path)
        print(f"Model saved to: {large_model_path}")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        # Save the plot
        history_path = os.path.join(base_dir, "training_history.png")
        plt.savefig(history_path)
        print(f"Training history saved to: {history_path}")
        
        # Return the model path
        return large_model_path
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        return None

if __name__ == "__main__":
    print("\n===== EMOTION RECOGNITION PERFORMANCE TEST =====")
    
    # Check if the user wants to test existing model or train on testdata
    train_new = input("Train new model on testdata? (y/n): ").lower() == 'y'
    
    if train_new:
        # Get training parameters
        epochs = int(input("Number of epochs (default 15): ") or 15)
        learning_rate = float(input("Learning rate (default 0.0001): ") or 0.0001)
        use_augmentation = input("Use data augmentation? (y/n, default y): ").lower() != 'n'
        
        # Train model on testdata
        model_path = train_on_testdata(epochs, learning_rate, use_augmentation)
        
        if model_path and os.path.exists(model_path):
            # Test the newly trained model
            test_model(model_path, test_data_dir)
        else:
            print("Training failed or model not saved.")
    else:
        # Test the existing model
        if os.path.exists(model_path):
            test_model(model_path, test_data_dir)
        else:
            print(f"No existing model found at {model_path}")
            print("Please train a model first.")
    
    print("\n===== PERFORMANCE TEST COMPLETED =====") 