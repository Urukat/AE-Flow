# 1. Introduction 

AE-Flow is a method proposed by Y. Zhao, Q. Ding, and X. Zhang in the paper [AE-FLOW: AUTOENCODERS WITH NORMALIZING FLOWS FOR MEDICAL IMAGES ANOMALY DETECTION](https://openreview.net/forum?id=9OmCr1q54Z), and is used for anomaly detection in medical images. It combines autoencoder and a normalizing flow bottleneck to improve the accuracy and interpretability of anomaly detection. The method provides not only image-level computability for normal data, but also pixel-level interpretability for anomalous data. Experiments conducted on different medical image datasets show the effectiveness and robustness of AE-FLOW, which has a large room for improvement in terms of anomaly detection compared with other relevant and representative methods. In this project, we firstly reproduce the original AE-FLOW model, studied its architecture, and finally made novel changes to its structure to improve the performance.

The AE-FLOW model follows an encoder-flow-decoder achitecture.

### The following are the key components of AE-Flow:

1. ##Encoder##: The encoder block takes an input image x and extracts low-dimensional features z using a function f : X → Z. In the AE-FLOW model, the encoder transforms the input image x ∈ R<sup>3×H×W</sup> to a feature z ∈ R<sup>C×$$/frac{H}{16}$$×W/16</sup>, where H and W are the height and width of the original image, C is the number of channels, and 16 is a downsampling factor.
![e-f-d](encoder-decoder.png)

2. ##Normalizing flows##: The Normalizing Flow Model is a generative model that converts a complex distribution into a simple distribution. In AE-Flow, the regularizing flow model is used to convert the feature vectors extracted by the autoencoder into a standard Gaussian distribution. The normalizing flows transform the feature vector z to a standard Gaussian distribution. This is done by applying a series of invertible transformations to z that preserve its dimensionality and allow for efficient computation of likelihoods. The likelihood of the normalized feature is used as the flow loss in training.

3. ##Decoder##: The decoder block reconstructs the normalized feature z' to an output image x' using a function g : Z → X'. In the AE-FLOW model, the decoder takes the normalized feature z' and produces an output image x' ∈ R3×H×W that has similar appearance to the input image x. The residual between x and x' is used as the reconstruction loss in training.

![architecture](architecture.png)

Overall, this encoder-flow-decoder architecture allows for efficient learning of low-dimensional representations of images that can be used for tasks such as anomaly detection or image generation.


### Loss Function

Two loss functions are used in AE-Flow to train the model. The first loss function is the Flow Loss, which measures the degree of anomaly based on the similarity between the standard Gaussian distribution and the transformed feature vectors. The second loss function is Reconstruction Loss, which measures the reconstruction quality based on the difference between the original image and the reconstructed image.

Unlike other flow models, AE-Flow can provide a certain degree of interpretability. By comparing the original image, the reconstructed image and the residual image, a better understanding of how the model detects anomalies can be obtained.

### Related Works

# 2. Weaknesses/Strengths/Potential

# 3. Novel Contribution

# 4. Results

# 5. Conclusion

# 6. Contribution
