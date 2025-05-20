
Program Overview

This program builds an advanced product recommendation system that integrates:

Vision Transformer (ViT) for analyzing product images
Neural network for processing numerical product information
Multi-Armed Bandits (MAB) algorithm to improve recommendations over time

Step 1: Import Necessary Libraries
The program uses several libraries:

pandas and numpy for data processing
matplotlib and seaborn for visualization
scikit-learn for data processing and evaluation
torch and related modules for deep learning
timm to provide pre-trained Vision Transformer models
PIL and requests for handling and downloading images

Step 2: Read and Explore Data
The program reads data from the file 'Data_Clean.csv' and displays:

Data size
Structural information
Descriptive statistics
First 5 rows

Step 3: Data Preprocessing
The program performs:

Removal of rows with null values in critical columns
Creation of new features:
discount_percentage: percentage of discount
price_per_rating: price per average rating
Display of the distribution of numerical features

Step 4: Create Dataset Class for Handling Numerical and Image Data
The ProductDataset class is defined to:

Process numerical data by standardizing with StandardScaler
Handle images from URLs, cache them, and apply transformations
Return a tuple (numerical features, image, label, index) for each sample

Step 5: Build Hybrid Model Combining Vision Transformer and Numerical Features
The HybridRecommendationModel class integrates:

Vision Transformer (ViT) for image processing
Neural network for numerical feature processing
Concatenation of features and prediction through another neural network Detailed structure:
ViT extracts features from images (768 features)
Neural network processes 7 numerical features, producing 128 features
Concatenation of the two feature sets (768 + 128 = 896)
Final prediction network with 3 layers (896 → 256 → 64 → 1)

Step 6: Implement Multi-Armed Bandits (MAB) Algorithm
The EpsilonGreedyMAB class implements the ε-greedy MAB algorithm:

Initialized with num_arms as the number of choices
select_arm(): randomly selects an arm with probability ε, otherwise chooses the best arm
update(arm, reward): updates the value of an arm based on the reward This algorithm balances exploitation and exploration.

Step 7: Set Up Training Process
The train_model function trains the hybrid model:

Uses GPU if available
Runs through multiple epochs with standard training and evaluation processes
Saves the best model based on validation loss
Plots the loss during training

Step 8: Build Evaluation and Model Comparison Function
The evaluate_recommendations function evaluates model performance:

Can use MAB or not
Calculates RMSE (Root Mean Squared Error)
Returns detailed prediction results The ContentBasedModel class implements a content-based recommendation method:
Uses similarity to find similar products
Predicts based on the average price of the 5 most similar products

Step 9: Set Up Main Pipeline
The main() function organizes the entire workflow:

Read and preprocess data
Split into training/validation/test sets
Initialize and train the hybrid model
Evaluate models:
Hybrid ViT + NN + MAB
Hybrid ViT + NN without MAB
Content-Based
Compare results and generate plots
Analyze recommendations and save results

Step 10: Execute Code and Display Results
Call the main() function and summarize results:

RMSE results for each model
Mean absolute error
List of generated files
Notable Aspects of the Program
Multimodal Hybrid Architecture: Combines image features (ViT) and numerical features (Neural Network) for more comprehensive predictions.
Pre-trained ViT Utilization: Leverages transfer learning to use knowledge from large datasets.
Optimization with MAB: Uses the MAB algorithm to automatically adjust recommendation strategies based on feedback.
Image Caching: Improves performance by storing downloaded images.
Comparison with Baseline: Evaluates the hybrid model against a traditional content-based method.
