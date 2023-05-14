# 1. Introduction 
AE-Flow is a method proposed by Y. Zhao, Q. Ding, and X. Zhang in the paper [AE-FLOW: AUTOENCODERS WITH NORMALIZING FLOWS FOR MEDICAL IMAGES ANOMALY DETECTION](https://openreview.net/forum?id=9OmCr1q54Z), and is used for anomaly detection in medical images. It combines autoencoder and a normalizing flow bottleneck to improve the accuracy and interpretability of anomaly detection. The method provides not only image-level computability for normal data, but also pixel-level interpretability for anomalous data. Experiments conducted on different medical image datasets show the effectiveness and robustness of AE-FLOW, which has a large room for improvement in terms of anomaly detection compared with other relevant and representative methods. In this project, we firstly reproduce the original AE-FLOW model, studied its architecture, and finally made novel changes to its structure to improve the performance.

### The following are the key components of AE-Flow:

1. Autoencoder: Autoencoder is a neural network model that compresses the input data into a low-dimensional space and reconstructs the original data from it. In AE-Flow, Autoencoder is used to extract the low-dimensional features of the input image.

2. Normalizing Flow Model: The Normalizing Flow Model is a generative model that converts a complex distribution into a simple distribution. In AE-Flow, the regularizing flow model is used to convert the feature vectors extracted by the autoencoder into a standard Gaussian distribution.

3. Loss Function: Two loss functions are used in AE-Flow to train the model. The first loss function is the Flow Loss, which measures the degree of anomaly based on the similarity between the standard Gaussian distribution and the transformed feature vectors. The second loss function is Reconstruction Loss, which measures the reconstruction quality based on the difference between the original image and the reconstructed image.

4. Interpretability: Unlike other flow models, AE-Flow can provide a certain degree of interpretability. By comparing the original image, the reconstructed image and the residual image, a better understanding of how the model detects anomalies can be obtained.

### Related Works

# 2. Weaknesses/Strengths/Potential

# 3. Novel Contribution

# 4. Results

# 5. Conclusion

# 6. Contribution
