#Synth-chimera

The Synth-chimera project aims to create a multimodal dataset for studying feature selection using genetic algorithms and Particle Swarm Optimization, utilizing a convolutional neural network as the fitness function. Also includes automatic selection of the processing unit, whether it be a GPU, Macbook's MPS, or CPU.

The MultimodalSyntheticDataset class is a custom dataset designed for generating synthetic multimodal data, including structured data and corresponding images. This dataset is particularly useful for machine learning tasks that require both types of data.

*Key Features*:
   Initialization Parameters:

   num_samples: Total number of samples in the dataset.

   num_features: Number of features for the structured data.

   image_size: Size of the generated images.

   num_classes: Number of distinct classes in the dataset.

*Data Generation*:

   *Structured Data*: Generated within specific ranges for each feature and label. The features are divided into two groups: the first half has values within class-specific ranges, aims to create random values within established patterns by class and feature, as simulating laboratory test indicators, while the second half has values within a broader range, creating noisy features.

   *Image Data*: Synthetic images are generated with simple patterns and colors based on the label. Each image contains a circle whose size and color are determined by the label.