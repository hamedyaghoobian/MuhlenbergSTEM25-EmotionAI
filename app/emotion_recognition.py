import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
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
import traceback

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition - STEM Day")
        self.root.geometry("1200x700")
        
        # Create accent style for important buttons
        style = ttk.Style()
        style.configure("Accent.TButton", 
                        background="#ff5252", 
                        foreground="white", 
                        font=('Arial', 10, 'bold'))
        
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
        
        # Add reset button
        self.reset_btn = ttk.Button(self.webcam_control_frame, text="Reset Everything", command=self.reset_application)
        self.reset_btn.pack(side=tk.RIGHT, padx=5)
        self.reset_btn.configure(style="Accent.TButton")
        
        # Emotion selection and capture
        self.emotion_frame = ttk.LabelFrame(self.right_frame, text="Capture Training Images", padding=10)
        self.emotion_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.emotion_var = tk.StringVar(value=self.emotions[0])
        self.emotion_dropdown = ttk.Combobox(self.emotion_frame, textvariable=self.emotion_var, values=self.emotions)
        self.emotion_dropdown.pack(fill=tk.X, padx=5, pady=5)
        
        self.capture_btn = ttk.Button(self.emotion_frame, text="Capture Image", command=self.capture_image)
        self.capture_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Add explanation button
        self.explain_capture_btn = ttk.Button(self.emotion_frame, text="How Capture Works", 
                                             command=lambda: self.show_explanation("capture"))
        self.explain_capture_btn.pack(fill=tk.X, padx=5, pady=5)
        
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
        
        # Add parameter controls
        self.param_frame = ttk.Frame(self.training_frame)
        self.param_frame.pack(fill=tk.X, pady=5)
        
        # Epochs parameter
        epoch_frame = ttk.Frame(self.param_frame)
        epoch_frame.pack(fill=tk.X, pady=2)
        ttk.Label(epoch_frame, text="Epochs:").pack(side=tk.LEFT)
        self.epoch_var = tk.IntVar(value=10)
        epochs_spin = ttk.Spinbox(epoch_frame, from_=1, to=20, textvariable=self.epoch_var, width=5)
        epochs_spin.pack(side=tk.RIGHT)
        ttk.Label(epoch_frame, text="(Higher = More training time)").pack(side=tk.RIGHT, padx=5)
        
        # Learning rate parameter
        lr_frame = ttk.Frame(self.param_frame)
        lr_frame.pack(fill=tk.X, pady=2)
        ttk.Label(lr_frame, text="Learning Rate:").pack(side=tk.LEFT)
        self.lr_var = tk.DoubleVar(value=0.0001)
        lr_combo = ttk.Combobox(lr_frame, textvariable=self.lr_var, width=8, 
                                values=[0.01, 0.001, 0.0001, 0.00001])
        lr_combo.pack(side=tk.RIGHT)
        ttk.Label(lr_frame, text="(Lower = More precise)").pack(side=tk.RIGHT, padx=5)
        
        # Data augmentation parameter
        aug_frame = ttk.Frame(self.param_frame)
        aug_frame.pack(fill=tk.X, pady=2)
        ttk.Label(aug_frame, text="Data Augmentation:").pack(side=tk.LEFT)
        self.aug_var = tk.BooleanVar(value=True)
        aug_check = ttk.Checkbutton(aug_frame, variable=self.aug_var)
        aug_check.pack(side=tk.RIGHT)
        ttk.Label(aug_frame, text="(Artificially expands training data)").pack(side=tk.RIGHT, padx=5)
        
        # Train button
        self.train_btn = ttk.Button(self.training_frame, text="Train Model", command=self.train_model)
        self.train_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Add explanation button
        self.explain_train_btn = ttk.Button(self.training_frame, text="How Training Works", 
                                           command=lambda: self.show_explanation("training"))
        self.explain_train_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress = ttk.Progressbar(self.training_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, padx=5, pady=5)
        
        # Testing controls
        self.testing_frame = ttk.LabelFrame(self.right_frame, text="Test Your Model", padding=10)
        self.testing_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.predict_btn = ttk.Button(self.testing_frame, text="Predict Emotion", command=self.predict_emotion)
        self.predict_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Add explanation button
        self.explain_predict_btn = ttk.Button(self.testing_frame, text="How Prediction Works", 
                                             command=lambda: self.show_explanation("prediction"))
        self.explain_predict_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.result_label = ttk.Label(self.testing_frame, text="Result: None")
        self.result_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Status bar - create a simple status bar with a border and background color
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, 
                                   bg="#e0e0e0", font=('Arial', 10), height=1, padx=10)
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
    
    class ProgressCallback(tf.keras.callbacks.Callback):
        """Callback for updating the progress bar during training."""
        def __init__(self, app):
            super().__init__()
            self.app = app
        
        def on_epoch_end(self, epoch, logs=None):
            # Calculate progress
            if self.params.get('epochs'):
                progress_value = int((epoch + 1) / self.params['epochs'] * 100)
            else:
                progress_value = 0
                
            # Get metrics
            acc = logs.get('accuracy', 0) * 100
            val_acc = logs.get('val_accuracy', 0) * 100
            
            # Update UI from main thread
            self.app.root.after(0, lambda: self.app.update_progress(
                progress_value, 
                f"Epoch {epoch+1}/{self.params['epochs']} - accuracy: {acc:.1f}% - val_accuracy: {val_acc:.1f}%"))
    
    def _train_model_thread(self):
        """Train the model in a separate thread."""
        try:
            # Get parameters from UI
            epochs = self.epoch_var.get()
            learning_rate = self.lr_var.get()
            use_augmentation = self.aug_var.get()
            
            # Update status
            self.status_bar.config(text=f"Training with {epochs} epochs, learning rate {learning_rate}, data augmentation: {use_augmentation}")
            
            # Set up data augmentation
            if use_augmentation:
                # More aggressive data augmentation for small datasets
                train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=30,  # Increased rotation range
                    width_shift_range=0.2,  # Increased shift range
                    height_shift_range=0.2,  # Increased shift range
                    shear_range=0.2,  # Increased shear range
                    zoom_range=0.3,  # More aggressive zoom
                    horizontal_flip=True,
                    brightness_range=[0.7, 1.3],  # Add brightness variation
                    fill_mode='nearest',
                    validation_split=0.2  # 20% validation split
                )
            else:
                train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    validation_split=0.2
                )
            
            # Check that we have data for all emotions
            empty_classes = []
            for emotion in self.emotions:
                emotion_dir = os.path.join(self.data_dir, emotion)
                if not os.path.exists(emotion_dir) or len(os.listdir(emotion_dir)) < 3:
                    empty_classes.append(emotion)
            
            if empty_classes:
                self.root.after(0, lambda: messagebox.showerror(
                    "Insufficient Data", 
                    f"Not enough images for: {', '.join(empty_classes)}.\nCapture at least 3 images for each emotion."
                ))
                self.train_btn.config(state=tk.NORMAL)
                self.progress['value'] = 0
                self.is_training = False
                return
            
            # Create data generators
            train_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=(224, 224),
                batch_size=8,  # Smaller batch size for small datasets
                class_mode='categorical',
                subset='training'
            )
            
            validation_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=(224, 224),
                batch_size=8,  # Smaller batch size for small datasets
                class_mode='categorical',
                subset='validation'
            )
            
            # Set up model
            base_model = MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Create model with regularization
            model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                Dropout(0.6),  # Increased dropout for better generalization
                Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                Dropout(0.5),
                Dense(len(self.emotions), activation='softmax')
            ])
            
            # Callbacks for better training
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                self.ProgressCallback(self)
            ]
            
            # Compile model with specified learning rate
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                callbacks=callbacks
            )
            
            # Save model
            model_path = os.path.join(self.models_dir, "emotion_model.h5")
            model.save(model_path)
            
            # Store the trained model
            self.model = model
            
            # Show training results
            self.training_complete(history)
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            traceback.print_exc()
            self.root.after(0, lambda: self.status_bar.config(text="Training failed!"))
            self.root.after(0, lambda: messagebox.showerror("Training Error", f"Error training model: {str(e)}"))
            self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL))
            self.is_training = False
    
    def update_progress(self, value, status_text):
        self.progress['value'] = value
        self.status_bar.config(text=status_text)
    
    def training_complete(self, history):
        self.is_training = False
        self.train_btn.config(state=tk.NORMAL)
        self.status_bar.config(text="Training completed!")
        
        # Show a message with the final accuracy
        final_acc = history.history['accuracy'][-1] * 100
        final_val_acc = history.history['val_accuracy'][-1] * 100
        
        # Create a training history window
        history_window = tk.Toplevel(self.root)
        history_window.title("Training Results")
        history_window.geometry("600x500")
        
        # Create a figure for plotting
        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot training & validation accuracy
        ax.plot(history.history['accuracy'], label='Training')
        ax.plot(history.history['val_accuracy'], label='Validation')
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(loc='lower right')
        ax.grid(True)
        
        # Embed the figure in the window
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=history_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1, padx=10, pady=10)
        
        # Add explanation text
        explanation_text = f"""
Training Results:

• Final Training Accuracy: {final_acc:.1f}%
• Final Validation Accuracy: {final_val_acc:.1f}%

What This Means:
• Training accuracy: How well the model learned the training images
• Validation accuracy: How well it generalizes to unseen images

Good Model:
• Both accuracies are high
• Small gap between training and validation accuracy

Potential Issues:
• Low accuracy: Need more training data or adjustments
• Big gap: Model might be "memorizing" rather than learning
        """
        
        text_frame = ttk.Frame(history_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, height=12)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, explanation_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Add close button
        close_btn = ttk.Button(history_window, text="Close", command=history_window.destroy)
        close_btn.pack(pady=10)
        
        messagebox.showinfo("Success", f"Model training completed!\nAccuracy: {final_acc:.1f}%\nValidation Accuracy: {final_val_acc:.1f}%")
    
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
                    self.status_bar.config(text="Loading saved model...")
                    self.model = load_model(model_path)
                    self.status_bar.config(text="Model loaded successfully")
                except Exception:
                    messagebox.showerror("Error", "No trained model available. Please train a model first.")
                    return
            else:
                messagebox.showerror("Error", "No trained model available. Please train a model first.")
                return
        
        # Preprocess the current frame for prediction
        self.status_bar.config(text="Processing image for prediction...")
        frame = cv2.flip(self.current_frame, 1)  # Mirror effect
        resized_img = cv2.resize(frame, (224, 224))
        preprocessed_img = preprocess_input(np.expand_dims(resized_img, axis=0))
        
        # Make prediction
        self.status_bar.config(text="Making prediction...")
        predictions = self.model.predict(preprocessed_img)
        
        # Create details window for the prediction
        details_window = tk.Toplevel(self.root)
        details_window.title("Prediction Details")
        details_window.geometry("500x400")
        
        # Display the image being analyzed
        img_frame = ttk.Frame(details_window)
        img_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Convert and resize for display
        display_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        display_img = Image.fromarray(display_img)
        display_img = ImageTk.PhotoImage(display_img)
        
        # Display the image
        img_label = ttk.Label(img_frame, image=display_img)
        img_label.image = display_img  # Keep a reference
        img_label.pack()
        
        # Create a frame for the predictions
        pred_frame = ttk.Frame(details_window)
        pred_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        ttk.Label(pred_frame, text="Emotion Predictions", font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        
        # Create a bar chart of predictions
        for i, emotion in enumerate(self.emotions):
            prob = predictions[0][i] * 100
            
            # Create a frame for this emotion
            emotion_frame = ttk.Frame(pred_frame)
            emotion_frame.pack(fill=tk.X, pady=2)
            
            # Add the emotion label
            ttk.Label(emotion_frame, text=f"{emotion}:", width=10, anchor=tk.W).pack(side=tk.LEFT)
            
            # Add a progress bar for the probability
            prob_bar = ttk.Progressbar(emotion_frame, length=300, value=prob)
            prob_bar.pack(side=tk.LEFT, padx=5)
            
            # Add the percentage text
            ttk.Label(emotion_frame, text=f"{prob:.1f}%").pack(side=tk.LEFT)
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class] * 100
        predicted_emotion = self.emotions[predicted_class]
        
        # Add a separator
        ttk.Separator(details_window, orient='horizontal').pack(fill=tk.X, padx=10, pady=10)
        
        # Display the final result
        result_frame = ttk.Frame(details_window)
        result_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(result_frame, 
                 text=f"Predicted emotion: {predicted_emotion} ({confidence:.1f}%)",
                 font=('Arial', 12, 'bold')).pack()
                 
        # Close button
        ttk.Button(details_window, text="Close", command=details_window.destroy).pack(pady=10)
        
        # Update result label
        self.result_label.config(text=f"Result: {predicted_emotion} ({confidence:.1f}%)")
        
        # Update status
        self.status_bar.config(text=f"Predicted emotion: {predicted_emotion}")

    def show_explanation(self, topic):
        """Show an explanation dialog about a specific topic."""
        explanations = {
            "training": """
How Model Training Works:

1. Data Collection: 
   - Your images are organized by emotion labels
   - Images are resized to 224x224 pixels
   
2. Data Augmentation (if enabled):
   - Creates variations of your images by:
     • Rotating slightly
     • Zooming in/out
     • Shifting position
     • Flipping horizontally
   - This helps the model generalize better with less data
   
3. Transfer Learning:
   - We start with MobileNetV2, a pre-trained model that already recognizes basic features
   - We add our own layers on top to recognize emotions
   
4. Training Process:
   - The model makes predictions on your images
   - It calculates how wrong it is (loss function)
   - It adjusts its parameters to be less wrong next time
   - This repeats for the number of epochs you set
   
5. Parameters You Can Change:
   - Epochs: More epochs = more learning cycles (may overfit)
   - Learning Rate: Controls how big the adjustments are
   - Data Augmentation: Artificially expands your training data
            """,
            
            "capture": """
How Image Capture Works:

1. When you click "Capture Image":
   - A frame is grabbed from your webcam
   - It's flipped horizontally (mirror effect)
   - It's resized to 224x224 pixels for the model
   
2. The image is labeled with your selected emotion
   
3. It's saved to the data directory under the emotion folder
   
4. Why capture multiple images?
   - ML models need examples to learn patterns
   - More diverse examples = better generalization
   - Try different expressions, angles, and lighting
            """,
            
            "prediction": """
How Emotion Prediction Works:

1. When you click "Predict Emotion":
   - A frame is grabbed from your webcam
   - It's processed the same way as training images
   
2. The image is passed through the neural network:
   - Feature extraction layers identify patterns
   - Final layers map these patterns to emotions
   
3. The output is a probability for each emotion
   - We show the emotion with highest probability
   - Confidence percentage shows how sure the model is
   
4. Common issues with predictions:
   - Low confidence: Model unsure between emotions
   - Wrong predictions: Need more training examples
   - Overfitting: Too specialized to your training data
            """
        }
        
        explanation_window = tk.Toplevel(self.root)
        explanation_window.title(f"Understanding {topic.title()}")
        explanation_window.geometry("600x500")
        
        # Add explanation text
        text_widget = tk.Text(explanation_window, wrap=tk.WORD, padx=15, pady=15)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, explanations.get(topic, "No explanation available for this topic."))
        text_widget.config(state=tk.DISABLED)  # Make read-only
        
        # Add close button
        close_btn = ttk.Button(explanation_window, text="Close", command=explanation_window.destroy)
        close_btn.pack(pady=10)

    def reset_application(self):
        """Reset the application by clearing all training data and model."""
        if messagebox.askyesno("Reset Application", 
                               "This will delete all your training images and reset the application.\n\nAre you sure?"):
            try:
                # Delete all captured images
                for emotion in self.emotions:
                    emotion_dir = os.path.join(self.data_dir, emotion)
                    if os.path.exists(emotion_dir):
                        for file in os.listdir(emotion_dir):
                            if file.endswith(".jpg"):
                                os.remove(os.path.join(emotion_dir, file))
                
                # Delete trained model
                model_path = os.path.join(self.models_dir, "emotion_model.h5")
                if os.path.exists(model_path):
                    os.remove(model_path)
                
                # Reset counters
                self.capture_counts = {emotion: 0 for emotion in self.emotions}
                for emotion in self.emotions:
                    self.count_labels[emotion].config(text="0 images")
                
                # Reset model
                self.model = None
                
                # Reset result
                self.result_label.config(text="Result: None")
                
                # Reset progress bar
                self.progress['value'] = 0
                
                # Update status
                self.status_bar.config(text="Reset complete! Ready for a fresh start.")
                
                messagebox.showinfo("Reset Complete", "All training data and model have been cleared.\nThe application is ready for a fresh start.")
            
            except Exception as e:
                messagebox.showerror("Error", f"Error resetting application: {str(e)}")

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