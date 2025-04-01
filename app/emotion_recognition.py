import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import matplotlib.pyplot as plt

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition - STEM Day")
        self.root.geometry("1200x700")
        
        # Emotion classes
        self.emotions = ["Happy", "Sad", "Surprised", "Neutral"]
        
        # Data directories
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data")
        self.models_dir = os.path.join(self.base_dir, "models")
        
        # Create directories for each emotion
        for emotion in self.emotions:
            os.makedirs(os.path.join(self.data_dir, emotion), exist_ok=True)
        
        # Initialize webcam
        self.cap = None
        self.is_webcam_on = False
        self.current_frame = None
        
        # Initialize UI components
        self.create_ui()
        
        # Model
        self.model = None
        self.is_training = False
        
        # Capture counters
        self.capture_counts = {emotion: 0 for emotion in self.emotions}
        
    def create_ui(self):
        # Main frames
        self.left_frame = ttk.Frame(self.root, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.right_frame = ttk.Frame(self.root, padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        # Webcam display
        self.webcam_label = ttk.Label(self.left_frame)
        self.webcam_label.pack(padx=10, pady=10)
        
        # Webcam control buttons
        self.webcam_control_frame = ttk.Frame(self.left_frame)
        self.webcam_control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_webcam_btn = ttk.Button(self.webcam_control_frame, text="Start Webcam", command=self.toggle_webcam)
        self.start_webcam_btn.pack(side=tk.LEFT, padx=5)
        
        # Emotion selection and capture
        self.emotion_frame = ttk.LabelFrame(self.right_frame, text="Capture Training Images", padding=10)
        self.emotion_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.emotion_var = tk.StringVar(value=self.emotions[0])
        self.emotion_dropdown = ttk.Combobox(self.emotion_frame, textvariable=self.emotion_var, values=self.emotions)
        self.emotion_dropdown.pack(fill=tk.X, padx=5, pady=5)
        
        self.capture_btn = ttk.Button(self.emotion_frame, text="Capture Image", command=self.capture_image)
        self.capture_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Image count display
        self.count_frame = ttk.Frame(self.right_frame)
        self.count_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.count_labels = {}
        for emotion in self.emotions:
            frame = ttk.Frame(self.count_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{emotion}:").pack(side=tk.LEFT)
            self.count_labels[emotion] = ttk.Label(frame, text="0 images")
            self.count_labels[emotion].pack(side=tk.RIGHT)
        
        # Training controls
        self.training_frame = ttk.LabelFrame(self.right_frame, text="Model Training", padding=10)
        self.training_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.train_btn = ttk.Button(self.training_frame, text="Train Model", command=self.train_model)
        self.train_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress = ttk.Progressbar(self.training_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
        # Testing controls
        self.testing_frame = ttk.LabelFrame(self.right_frame, text="Test Your Model", padding=10)
        self.testing_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.predict_btn = ttk.Button(self.testing_frame, text="Predict Emotion", command=self.predict_emotion)
        self.predict_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.result_label = ttk.Label(self.testing_frame, text="Result: None")
        self.result_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Status bar
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def toggle_webcam(self):
        if self.is_webcam_on:
            self.is_webcam_on = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.start_webcam_btn.config(text="Start Webcam")
        else:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return
            
            self.is_webcam_on = True
            self.start_webcam_btn.config(text="Stop Webcam")
            self.update_webcam()
    
    def update_webcam(self):
        if self.is_webcam_on and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                # Flip horizontally for a mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                # Resize while maintaining aspect ratio
                target_width = 640
                w_percent = target_width / float(pil_img.width)
                target_height = int(float(pil_img.height) * w_percent)
                pil_img = pil_img.resize((target_width, target_height), Image.LANCZOS)
                
                # Convert to Tkinter-compatible photo image
                img_tk = ImageTk.PhotoImage(image=pil_img)
                
                # Update the image in label
                self.webcam_label.configure(image=img_tk)
                self.webcam_label.image = img_tk  # Keep a reference
            
            # Call this method again after 10ms
            self.root.after(10, self.update_webcam)
    
    def capture_image(self):
        if self.current_frame is None:
            messagebox.showwarning("Warning", "Webcam is not active")
            return
        
        selected_emotion = self.emotion_var.get()
        save_dir = os.path.join(self.data_dir, selected_emotion)
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the image
        count = self.capture_counts[selected_emotion]
        filename = f"{selected_emotion}_{count+1}.jpg"
        save_path = os.path.join(save_dir, filename)
        
        # Preprocess image for saving (resize to 224x224)
        frame = cv2.flip(self.current_frame, 1)  # Mirror effect
        resized_img = cv2.resize(frame, (224, 224))
        cv2.imwrite(save_path, resized_img)
        
        # Update counter
        self.capture_counts[selected_emotion] += 1
        self.count_labels[selected_emotion].config(text=f"{self.capture_counts[selected_emotion]} images")
        
        # Update status
        self.status_bar.config(text=f"Captured {filename}")
    
    def train_model(self):
        if self.is_training:
            return
        
        # Check if we have enough images
        min_count = min(self.capture_counts.values())
        if min_count < 5:
            messagebox.showwarning("Warning", "Please capture at least 5 images for each emotion")
            return
        
        # Start training in a separate thread
        self.is_training = True
        self.train_btn.config(state=tk.DISABLED)
        self.status_bar.config(text="Training started...")
        self.progress['value'] = 0
        
        thread = threading.Thread(target=self._train_model_thread)
        thread.daemon = True
        thread.start()
    
    def _train_model_thread(self):
        try:
            # Create data generator for data augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2  # 20% for validation
            )
            
            # Load training data
            train_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                subset='training'
            )
            
            # Load validation data
            validation_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                subset='validation'
            )
            
            # Create model using transfer learning with MobileNetV2
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            
            # Freeze base model layers
            for layer in base_model.layers:
                layer.trainable = False
            
            # Create new model on top
            self.model = Sequential([
                base_model,
                Conv2D(32, (3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(len(self.emotions), activation='softmax')
            ])
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Progress callback
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, app):
                    super().__init__()
                    self.app = app
                
                def on_epoch_end(self, epoch, logs=None):
                    progress_value = int((epoch + 1) / epochs * 100)
                    
                    # Update UI from main thread
                    self.app.root.after(0, lambda: self.app.update_progress(progress_value, f"Training: epoch {epoch+1}/{epochs}"))
            
            # Train model
            epochs = 10
            history = self.model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // train_generator.batch_size,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // validation_generator.batch_size,
                epochs=epochs,
                callbacks=[ProgressCallback(self)]
            )
            
            # Save model
            os.makedirs(self.models_dir, exist_ok=True)
            model_path = os.path.join(self.models_dir, "emotion_model.h5")
            self.model.save(model_path)
            
            # Update UI from main thread
            self.root.after(0, lambda: self.training_complete(history))
            
        except Exception as e:
            # Handle exceptions
            self.root.after(0, lambda: self.training_error(str(e)))
    
    def update_progress(self, value, status_text):
        self.progress['value'] = value
        self.status_bar.config(text=status_text)
    
    def training_complete(self, history):
        self.is_training = False
        self.train_btn.config(state=tk.NORMAL)
        self.status_bar.config(text="Training completed!")
        messagebox.showinfo("Success", "Model training completed successfully!")
    
    def training_error(self, error_msg):
        self.is_training = False
        self.train_btn.config(state=tk.NORMAL)
        self.status_bar.config(text="Training failed!")
        messagebox.showerror("Error", f"Training failed: {error_msg}")
    
    def predict_emotion(self):
        if self.current_frame is None:
            messagebox.showwarning("Warning", "Webcam is not active")
            return
        
        if self.model is None:
            # Try to load a saved model
            model_path = os.path.join(self.models_dir, "emotion_model.h5")
            if os.path.exists(model_path):
                try:
                    self.model = load_model(model_path)
                except Exception:
                    messagebox.showerror("Error", "No trained model available. Please train a model first.")
                    return
            else:
                messagebox.showerror("Error", "No trained model available. Please train a model first.")
                return
        
        # Preprocess the current frame for prediction
        frame = cv2.flip(self.current_frame, 1)  # Mirror effect
        resized_img = cv2.resize(frame, (224, 224))
        preprocessed_img = preprocess_input(np.expand_dims(resized_img, axis=0))
        
        # Make prediction
        predictions = self.model.predict(preprocessed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        
        # Map index to emotion label
        predicted_emotion = self.emotions[predicted_class]
        
        # Update result label
        self.result_label.config(text=f"Result: {predicted_emotion} ({confidence:.1f}%)")
        
        # Update status
        self.status_bar.config(text=f"Predicted emotion: {predicted_emotion}")

def main():
    print("Starting Emotion Recognition Application...")
    print("TensorFlow version:", tf.__version__)
    print("OpenCV version:", cv2.__version__)
    print("NumPy version:", np.__version__)
    print("Creating Tkinter root window...")
    root = tk.Tk()
    print("Initializing application...")
    app = EmotionRecognitionApp(root)
    print("Starting main event loop...")
    root.mainloop()
    print("Application closed")

if __name__ == "__main__":
    print("Script started")
    main()
    print("Script ended") 