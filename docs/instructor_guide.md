# Instructor Guide: Emotion Recognition Activity

## Overview

This guide will help you prepare and facilitate the Emotion Recognition AI activity for your STEM day event. The activity is designed to introduce high school students to the concepts of AI, machine learning, and computer vision through a hands-on experience.

## Activity Goals

Students will:
- Understand the basic workflow of AI development
- Collect and label their own training data
- Train a simple neural network model
- Test the model with new inputs
- Discuss applications and implications of AI

## Preparation Checklist

### Before the Event (1-2 days prior)

- [ ] Set up and test the software on all 4 computers
- [ ] Ensure all computers have working webcams
- [ ] Run through the complete activity on each computer
- [ ] Print student handouts (one per student)
- [ ] Generate the neural network diagram by running `python app/create_diagram.py`

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
- Distribute the handouts

### Guided Activity (20 minutes)
1. **Data Collection (10 minutes)**
   - Help students start the application and activate the webcam
   - Guide them through capturing images for each emotion
   - Encourage them to capture varied expressions

2. **Model Training (5 minutes)**
   - While the model trains, explain the concept of neural networks
   - Use the diagram to illustrate how the model processes images
   - Point out the progress bar and explain what's happening

3. **Testing (5 minutes)**
   - Let students test their models with new expressions
   - Encourage them to try "tricking" the model
   - Have them observe differences between groups' results

### Discussion (5 minutes)
- Lead a brief discussion on:
  - How their models performed
  - Ways to improve accuracy
  - Real-world applications of this technology
  - Ethical considerations

## Facilitating Tips

- **Technical Support:** Be prepared to help with webcam issues or software problems
- **Engagement:** Ask questions throughout to keep students engaged
- **Time Management:** Keep an eye on the clock - data collection often takes longer than expected
- **Differentiation:** For advanced students, discuss more complex aspects like overfitting, data augmentation

## Discussion Questions

Here are some questions to stimulate discussion:

1. Why did your model make certain mistakes? How could you fix them?
2. How did the amount of training data affect your model's performance?
3. What other applications could use similar technology?
4. What privacy concerns might arise with emotion recognition technology?
5. How could bias affect these kinds of AI systems?

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