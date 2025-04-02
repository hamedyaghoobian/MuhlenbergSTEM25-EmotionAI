# Build Your Own Emotion Recognition AI

## Workshop Overview
In this activity, you will build an AI system that can recognize facial expressions. You'll capture training data, train a neural network model, and test it to see how well it recognizes different emotions!

## Learning Objectives
- Understand how supervised learning works through hands-on experience
- Learn about data collection for AI training
- Experience neural network training and parameter tuning
- Evaluate and test AI model performance
- Explore real-world applications of computer vision

## 30-Minute Activity Outline

### 1. Data Collection (10 minutes)
- Start the application and activate your webcam
- Select an emotion from the dropdown menu
- Make facial expressions matching the selected emotion
- Capture at least 5-10 images for each emotion category
- Repeat for all four emotions (Happy, Sad, Surprised, Neutral)

### 2. Model Training (8 minutes)
- Examine the training parameters:
  - **Epochs**: Controls how many times the model sees your training data
  - **Learning Rate**: Determines how quickly the model makes adjustments
  - **Data Augmentation**: Creates variations of your images to get more training examples
- Click "Train Model" to start the learning process
- While it's training, explore the "How Training Works" button
- When training completes, examine the accuracy graph

### 3. Testing Your Model (7 minutes)
- Make different facial expressions to test your model
- Click "Predict Emotion" to see what the AI thinks you're expressing
- Examine the confidence scores for each emotion
- Try to "trick" your model with ambiguous expressions

### 4. Discussion and Exploration (5 minutes)
- Which emotions were easiest/hardest for your model to recognize?
- How did changing the training parameters affect your results?
- What happened when you tried unusual expressions?
- How could you improve your model's performance?

## Tips for Working with Small Datasets

Since we only have time to collect a small number of images during this workshop, here are some strategies to get the best results:

### Collecting Better Training Data
1. **Variety is key**: Capture images with different angles, expressions, and lighting
2. **Clear expressions**: Make very distinct facial expressions for each emotion
3. **Balanced dataset**: Try to capture a similar number of images for each emotion
4. **Quality over quantity**: A few clear, varied images are better than many similar ones

### Optimizing Training Parameters
1. **Always use data augmentation**: This creates additional training samples from your images
2. **Start with 10 epochs**: This is usually enough for small datasets
3. **Learning rate of 0.0001**: This slower rate helps prevent overfitting
4. **Be patient**: With small datasets, accuracy might not be extremely high

### Improving Results
1. **Add more training data**: If your model struggles with a particular emotion, add more images
2. **Try different expressions**: If "Sad" is confused with "Neutral," try more pronounced expressions
3. **Restart if needed**: Use the "Reset Everything" button if results are poor and start over

## Practice Activities

### Basic: Build Your Emotion Recognizer
1. Capture at least 5 images of each emotion
2. Train your model with default settings
3. Test with new expressions

### Challenge: Optimize Your Model
1. Try different parameter combinations:
   - More epochs (15-20)
   - Different learning rates
   - With/without data augmentation
2. Compare the results on the accuracy graph
3. Find the best combination for your data

## How AI Works Behind the Scenes

### Data Collection
When you capture images, the app:
1. Takes a photo from your webcam
2. Resizes it to 224×224 pixels
3. Stores it in the appropriate emotion folder
4. These labeled images form your "training dataset"

### Neural Network Training
When you train your model:
1. The app uses transfer learning with a pre-trained neural network (MobileNetV2)
2. It adapts this network to recognize your specific facial expressions
3. The model learns patterns from your training images
4. The accuracy graph shows how the model improves over time

### Prediction Process
When you test your model:
1. Your current facial expression is captured
2. The image is processed identically to training images
3. The neural network analyzes the image
4. It outputs confidence scores for each emotion
5. The emotion with the highest score is the prediction

## Model Performance with Larger Datasets

In professional AI applications, models are trained on thousands or millions of images. We tested our emotion recognition approach on a larger dataset with:
- 7,164 "Happy" images
- 4,938 "Sad" images 
- 3,205 "Surprised" images
- 4,982 "Neutral" images

The results were:
- Overall accuracy: ~65%
- Happy recognition: 85% accuracy
- Sad recognition: 47% accuracy
- Surprised recognition: 51% accuracy
- Neutral recognition: 69% accuracy

This demonstrates:
1. More data generally improves performance (compared to ~30 images per emotion)
2. Some emotions (like Happy) are easier to recognize than others
3. Even with thousands of images, emotion recognition remains challenging
4. Professional systems may need additional techniques beyond what we're using today

## Real-World Applications

Emotion recognition technology has many applications:
- Human-computer interaction improvements
- Customer satisfaction measurement
- Mental health monitoring
- Educational tools that adapt to student engagement
- Automotive systems that detect driver alertness

## Keep Exploring!

Want to learn more? Try:
- [Teachable Machine](https://teachablemachine.withgoogle.com/) - Another easy-to-use AI training tool
- [AI for Everyone](https://www.coursera.org/learn/ai-for-everyone) - A non-technical course about AI concepts
- [Machine Learning for Kids](https://machinelearningforkids.co.uk/) - More kid-friendly AI projects 