# STEM Day Emotion Recognition Activity

This project contains an interactive emotion recognition activity for high school students. Students will capture images of different facial expressions, train a simple neural network model, and test the model's ability to recognize emotions in real-time.

## Overview

This activity demonstrates the basic workflow of AI development:
1. **Data Collection**: Capturing and labeling facial expressions
2. **Model Training**: Training a neural network to recognize patterns
3. **Inference**: Using the trained model to make predictions on new data

## Prerequisites

- Python 3.8 or newer
- Webcam-enabled computer
- Basic understanding of Python (for instructors)

## Quick Setup

1. Clone this repository or download and extract the ZIP file
2. Open a command prompt or terminal
3. Navigate to the project directory
4. Run the setup script:

```bash
python setup.py
```

The setup script will:
- Check Python version
- Install required dependencies
- Create necessary directories
- Create a desktop shortcut

## Manual Setup

If the automatic setup doesn't work, you can manually install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

After setup, you can start the application by:

1. Double-clicking the desktop shortcut, or
2. Running the start script directly:

```bash
python app/start_app.py
```

## Activity Instructions for Students

1. **Start the application** from the desktop shortcut
2. **Click "Start Webcam"** to activate your webcam
3. **Capture training images**:
   - Select an emotion from the dropdown (Happy, Sad, Surprised, Neutral)
   - Make the corresponding facial expression
   - Click "Capture Image"
   - Repeat 10-15 times for each emotion
4. **Train your model**:
   - Click "Train Model"
   - Wait for training to complete (watch the progress bar)
5. **Test your model**:
   - Make a new facial expression
   - Click "Predict Emotion"
   - See if the model correctly identifies your emotion

## Technical Details

This application uses:
- TensorFlow/Keras for machine learning
- Transfer learning with MobileNetV2 (a pre-trained model)
- Tkinter for the graphical user interface
- OpenCV for webcam access and image processing

## Troubleshooting

- **Webcam not working**: Make sure no other application is using the webcam
- **Installation errors**: Make sure you have admin/sudo privileges
- **Training errors**: Ensure you've captured at least 5 images for each emotion

## License

This project is available for educational purposes. 