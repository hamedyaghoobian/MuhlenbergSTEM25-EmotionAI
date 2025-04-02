# Instructor Guide: Emotion Recognition Activity

## Overview

This guide will help you prepare and facilitate the enhanced Emotion Recognition AI activity for your STEM day event. The application now includes interactive explanations, parameter controls, and visualizations to help students understand what's happening "under the hood" during the AI development process.

## Activity Goals

Students will:
- Understand the basic workflow of AI development
- Collect and label their own training data
- Experiment with different model training parameters
- Visualize and interpret model training results
- Test the model with new inputs
- Discuss applications and implications of AI

## Preparation Checklist

### Before the Event (1-2 days prior)

- [ ] Set up and test the software on all 4 computers
- [ ] Ensure all computers have working webcams
- [ ] Run through the complete activity including the new parameter controls
- [ ] Print student handouts (one per student)
- [ ] Review the explanation dialogs so you can expand on these concepts

### Supplies Needed

- 4 computers with webcams
- Printed student handouts
- Optional: Whiteboard/flip chart for discussion

## Computer Setup Instructions

1. **System Requirements:**
   - Python 3.8 or newer
   - Working webcam
   - 4GB RAM minimum (8GB recommended)
   - Modern CPU (activity doesn't require GPU)

2. **Installation:**
   - Copy the `STEMEmotionRecognition` folder to each computer
   - Run the setup script:
     ```
     python setup.py
     ```
   - The script will install all dependencies and create a desktop shortcut
   - Test the application by launching it from the shortcut

3. **Troubleshooting Common Issues:**
   - **Webcam access denied**: Make sure the application has permission to access the webcam
   - **Dependency installation errors**: Try installing dependencies manually with `pip install -r requirements.txt`
   - **Application crashes**: Check console output for error messages

## Activity Timeline (30 minutes)

### Introduction (5 minutes)
- Welcome students and organize them into groups (1-2 students per computer)
- Give a brief overview of the activity and its connection to AI
- Highlight the new interactive features:
  - The "How it Works" explanation buttons
  - The parameter controls for experimentation
  - The visualization windows for predictions and training
- Distribute the handouts

### Guided Activity (20 minutes)
1. **Data Collection (8 minutes)**
   - Help students start the application and activate the webcam
   - Guide them through capturing images for each emotion
   - Encourage them to check the "How Capture Works" explanation
   - Have them try different expressions and angles
   - Point out the counter showing how many images they've captured
   - **Important**: Emphasize the need for at least 5 varied images per emotion

2. **Model Training (7 minutes)**
   - Introduce the parameter controls and what each one does:
     - **Epochs**: Number of training cycles (more = longer training)
     - **Learning Rate**: Size of adjustments (smaller = more precise)
     - **Data Augmentation**: Creating variations of their images
   - Have students try different parameter combinations (if time allows)
   - While the model trains, point out the progress bar showing accuracy
   - When training completes, discuss the accuracy graph
   - Have students observe differences between groups using different parameters

3. **Testing (5 minutes)**
   - Let students test their models with new expressions
   - Encourage them to examine the detailed prediction breakdown
   - Have them try the challenge activities like confusing the model
   - Compare results between groups

### Discussion (5 minutes)
- Lead a brief discussion on:
  - How different parameters affected model performance
  - Which emotions were easiest/hardest to recognize
  - What strategies improved accuracy
  - Real-world applications of this technology
  - Ethical considerations

## Small Dataset Optimizations

The application has been specially optimized to work with small datasets collected during the workshop:

### Technical Optimizations
- **Enhanced data augmentation**: Creates more variations from each training image
- **Regularization**: Prevents overfitting to the small training set
- **Early stopping**: Automatically stops training when performance stops improving
- **Smaller batch size**: Better suited for small datasets
- **Input validation**: Ensures students have at least 3 images per emotion

### Teaching Tips
- Remind students that professional AI systems use thousands or millions of images
- Explain that the app uses "transfer learning" to work with limited data
- If students are getting poor results, suggest:
  - Taking more varied photos (different angles, expressions)
  - Making more exaggerated expressions
  - Ensuring even lighting on their face
  - Using the "Reset Everything" button to start fresh

### Common Issues
- **"Insufficient Data" error**: Make sure students capture at least 3 images for each emotion
- **One emotion dominates predictions**: This indicates an imbalanced training set
- **Low accuracy**: This is normal with small datasets - focus on the learning process
- **Slow training**: This is expected due to the optimizations for small datasets

## New Interactive Features to Highlight

### Explanation Buttons
- **How Capture Works**: Explains image preprocessing and storage
- **How Training Works**: Details the training process and parameters
- **How Prediction Works**: Shows how the model makes decisions

### Parameter Controls
- Encourage students to experiment with these values
- Discuss the tradeoffs of different settings:
  - More epochs: Better accuracy but risk of overfitting
  - Higher learning rate: Faster training but may miss optimal values
  - Data augmentation: More training variety from fewer real examples

### Visualization Windows
- **Training Results**: Shows accuracy over time with explanation
- **Prediction Details**: Shows confidence scores for each emotion

## Facilitating Tips

- **Guided Experimentation:** Suggest specific parameter combinations for students to try
- **Comparisons:** Have groups compare results using different settings
- **Visual Learning:** Use the explanations and visualizations to reinforce concepts
- **Timeboxing:** The data collection phase often runs long - keep groups moving
- **Differentiation:** For advanced students, ask deeper questions about the visualization results

## Discussion Questions

Here are some enhanced questions to stimulate discussion:

1. How did different parameter settings affect your model's performance?
2. What patterns did you notice in the prediction confidence scores?
3. When would a higher or lower learning rate be beneficial?
4. Why might a model perform well on training data but poorly on new inputs?
5. What privacy or ethical concerns might arise with emotion recognition technology?
6. How could you collect better training data to improve your model?

## Follow-up Resources

For students who want to explore further:
- [Teachable Machine](https://teachablemachine.withgoogle.com/) - Similar but web-based tool
- [AI for Everyone](https://www.coursera.org/learn/ai-for-everyone) - Andrew Ng's introductory course
- [Machine Learning for Kids](https://machinelearningforkids.co.uk/) - Simplified ML projects

## Cleanup

After the activity:
1. Close all applications
2. If needed, uninstall dependencies:
   ```
   pip uninstall -r requirements.txt -y
   ```
3. Delete the application folder and desktop shortcut

## Technical Background

If students ask more advanced questions:

- **Transfer Learning:** We're using MobileNetV2 as a base model. It was pre-trained on ImageNet (a dataset of 1M+ images) and we're adapting it for emotion recognition.
- **CNN Architecture:** Convolutional Neural Networks use filters to detect features like edges, textures, and patterns in images.
- **Overfitting:** If a model performs well on training data but poorly on new data, it has "memorized" rather than "learned" general patterns.
- **Data Augmentation:** We artificially expand our dataset by applying transformations (rotation, zoom, etc.) to our original images.

## Technical Background for Advanced Questions

If students ask more advanced questions about the visualizations:

- **Training Graph**: The gap between training and validation accuracy indicates potential overfitting
- **Prediction Confidence**: Low confidence across all emotions may indicate poor feature extraction
- **Parameter Selection**: 
  - Epochs: More isn't always better (diminishing returns, overfitting)
  - Learning rate: Too high can cause overshooting, too low can get stuck in local minima
  - Data augmentation: Helps with small datasets but introduces artificial patterns

- **MobileNetV2 Architecture**: Uses depthwise separable convolutions for efficiency
- **Transfer Learning**: Only fine-tuning the top layers leverages pretrained feature extraction 