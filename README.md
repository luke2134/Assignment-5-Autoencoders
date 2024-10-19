# Assignment-5-Autoencoders


Analysis Report: Autoencoder for Olivetti Faces Dataset
Course: Unsupervised and Reinforcement Learning (COMP257)
Student Name: Ibrahem Aqel
Date: 2024-10-18

1. Introduction
The purpose of this assignment was to implement an autoencoder on the Olivetti faces dataset. An autoencoder is an unsupervised learning model used for compressing data by learning an encoded representation, and then decoding it back to its original form. We used PCA to reduce the dimensionality of the dataset and applied k-fold cross-validation to tune the model’s performance. This report analyzes the architecture, training process, and results of our model.

2. Dataset Preparation and PCA Analysis
The Olivetti faces dataset contains 400 grayscale images with 64×64 pixels each, leading to 4096 features per image. Training an autoencoder directly on such high-dimensional data would be computationally expensive. Thus, we applied PCA to reduce the dimensionality to 178 features, preserving 99% of the original variance. This allowed us to retain the key information while speeding up the training process.

After applying PCA, the dataset was split into 60% training, 20% validation, and 20% test sets. The stratified split ensured a balanced representation of each individual across the sets.

Insight:
The use of PCA effectively reduced the input size and computational cost without losing significant information. This ensured faster training and convergence during model optimization.

3. Autoencoder Architecture and Model Design
The autoencoder was designed to compress the data into a smaller representation (encoding) and then reconstruct it (decoding). Below is a summary of the architecture:

Encoder:

Input Layer: 178 units (PCA-reduced features)
Hidden Layer 1: 128 units, ReLU activation, L1 regularization
Bottleneck Layer: 64 units, ReLU activation
Decoder:

Hidden Layer 2: 128 units, ReLU activation
Output Layer: 178 units, Sigmoid activation
We used ReLU activation in hidden layers because it is computationally efficient and helps avoid the vanishing gradient problem. Sigmoid activation in the output layer ensured the reconstructed data remained normalized between 0 and 1. The Adam optimizer was chosen for fast and adaptive learning, and Mean Squared Error (MSE) was used as the loss function to measure reconstruction quality.

Insight:
The bottleneck layer with 64 units compressed the input data to a smaller representation, forcing the model to focus on the most important patterns in the images.

4. Training Process with K-Fold Cross-Validation
To ensure the model generalized well, we applied 5-fold cross-validation. This involved splitting the training data into 5 equal parts, with each part serving as a validation set once while the remaining 4 were used for training. This technique helped us select the best-performing model and avoid overfitting.

We trained the model for 50 epochs in each fold with a batch size of 32. After evaluating all 5 folds, the model with the lowest average validation loss was selected.

Best Validation Loss: 0.346
Insight:
Using cross-validation ensured that the model didn’t overfit to any specific subset of the data. It also provided a more reliable estimate of the model’s performance.

5. Test Results and Reconstruction Performance
The best model from cross-validation was evaluated on the test set to check its performance on unseen data. We achieved the following:

Test Loss: 0.318
We also visualized original vs. reconstructed images to assess the qualitative performance. The reconstructed images closely resembled the originals, with most of the key facial features intact. Some minor details, such as glasses and wrinkles, were slightly blurred in the reconstruction, which is expected due to the compression through the bottleneck layer.

Sample Output: Original vs. Reconstructed Images
Original Images	Reconstructed Images
Insight:
The reconstructed images demonstrate that the autoencoder successfully learned meaningful representations of the data. While the finer details are not perfect, the overall structure and key features were accurately captured.

6. Challenges and Limitations
While the autoencoder performed well overall, a few challenges were observed:

Loss of Fine Details: Some reconstructed images lost fine features, such as glasses and complex textures.
Dimensionality Reduction: PCA reduced the dimensionality significantly, which may have resulted in the loss of some important information.
Possible Improvements:

Increase the size of the bottleneck layer to capture more detailed patterns.
Use a convolutional autoencoder to better preserve spatial relationships in the images.
7. Conclusion
In this assignment, we successfully implemented an autoencoder for the Olivetti faces dataset. The use of PCA for dimensionality reduction improved training speed without sacrificing much information. The model achieved a test loss of 0.318, indicating that it generalized well to unseen data.

The reconstructed images closely resemble the originals, demonstrating that the autoencoder effectively learned compressed representations. With further improvements, such as increasing the bottleneck size or using convolutional layers, the model could achieve even better performance.

