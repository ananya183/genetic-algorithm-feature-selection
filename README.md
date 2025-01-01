# genetic-algorithm-feature-selection
Feature Selection for Few-Shot Building Instance Classification Using Genetic Algorithms in Artificial Intelligence

# Feature Selection for Few-Shot Building Instance Classification Using Genetic Algorithms

## Phase 1

### Abstract
In urban landscapes, buildings with both single-type and mixed-type usages are common. This project proposes the use of Genetic Algorithms and Contrastive Learning to classify building usage types from Street-View images. The first phase involved creating a dataset of mixed-type buildings using Google Street-View and evaluating two models, ResNet-50 and DINOv2, on a given dataset of single-type buildings. DINOv2 achieved superior results.

### Introduction
In countries like India, buildings often house various ventures on different floors, making mixed-type buildings harder to classify than single-type ones. This project aims to classify mixed-type buildings using Genetic Algorithms and Contrastive Learning, in addition to pre-existing models.

### Methodology
1. **Image Collection:** Google Street View was used to locate mixed-type buildings. Images were downloaded and converted from equirectangular to normal FOV format.
2. **Model Evaluation:** Two pre-trained models, ResNet-50 and DINOv2, trained on ImageNet, were evaluated on a dataset of single-type buildings across seven classes: Abandoned, Commercial, Industrial, Religious, Residential, Under-Construction, and Others.

### Results
- **ResNet-50:**
  - Accuracy: 58%
- **DINOv2:**
  - Accuracy: 86%

### Discussions
DINOv2 outperformed ResNet-50 due to its self-supervised learning capabilities and attention mechanism, which allows it to learn more robust and generalizable features from the data.

### Conclusion
DINOv2's better performance suggests it is more suitable for further phases of classifying mixed-type buildings in the Indian context.

## Phase 2

### Abstract

In the urban landscape, buildings with both a specified usage (i.e., single-type) and those that serve multiple purposes (i.e., mixed-type) are common. While it is easy to map single-type buildings, it proves difficult to do so for mixed types. This work proposes the use of Genetic Algorithms and Contrastive Learning to classify building usage types from Street-View images. For this phase, we continue our work from the past to train a model to get image embeddings of single-type images, from which class-wise centroids are calculated, which are then used to find the percentage composition of the mixed-type buildings. Triplet Loss was used for training purposes, and works like this, with a few more modifications, can be used for labelling mixed-type buildings to the class they have the maximum percentage composition of.

### Introduction

In developing countries like India, it is common to see a single building housing multiple ventures. The most common example of this would be buildings that have a grocery store, a commercial building type on the lower floors, and a residential part above these stores. Such buildings can be found all over India in various localities. Other similar examples include abandoned buildings that have survived several decades and are now being reused for commercial or residential purposes on the lower floors. Such buildings are common in the 'older' parts of Indian cities. Industrial buildings may have parts of them under construction. These are just a few examples of the different types of combinations that can be present in mixed-type buildings.

For this task, we have assumed that there are seven broad classes a single-type building can belong to: Abandoned, Commercial, Industrial, Religious, Residential, Under-Construction, and Others. A mixed-type building can belong to a combination of any of these categories. As such, marking the mixed-type buildings is arduous as it may not be immediately evident which category such buildings best fit into. As such, we decided to instead classify the category-wise percentage composition of mixed-type buildings from their Street-View images.

For this purpose, the task was divided into two phases. Phase 1 involved the creation of a dataset of mixed-type buildings and selecting a base model for the task of classifying single-type buildings into the seven categories defined earlier. This base model was then tuned for the task of getting the percentage composition of mixed-type buildings in the second phase. This was done through the use of, among many techniques, Genetic Algorithms and Triplet Loss.

### Brief Theoretical Overview

One important work done for building classification used Street-View images to classify the use type of single-type buildings and marked the building usage on an aerial view image of the building's locality. For this task, Street-View images were taken, outliers were removed using a pre-trained CNN on the Places2 dataset, and four models, ResNet18, ResNet34, AlexNet, and VGG16, were experimented with and fine-tuned for the classification task. VGG16 was finally selected based on the best F1 scores and accuracy.

Contrastive Learning is a Self-Supervised Learning paradigm used to combat the requirement for a high amount of labelled data for training CNNs. This method works similarly to kNN clustering in the sense that the data points that match are clustered closer while the negative matches are distanced. It allows the model to cluster the data points based on the high-level features and, thus, reduces the need for labelled data. Using this would allow the Street-View images directly, and Genetic Algorithms could be used in combination to allow faster and more precise clustering of the images.

## Methodology

### Creation of Triplets

Triplets of images were created. For each anchor image, a random image from the same category which was not the anchor image, was taken as the positive image, and a random image from another class was taken as the negative image. As such, we ended up with a total of 3772 triplets.

### Neural Network Implementation

For the purpose of getting the weights to implement for Genetic Algorithm, we implemented neural networks add-ons to the base DINOv2 model with 0, 1 and 2 hidden layers, which were each trained for 8 epochs and on different activation functions. For this step, various sizes of the hidden layers were tried, such as 256, 128, 512 and 64. Activation functions used included ReLU, LeakyReLu, tanh and GELU. Dropout was also used, with values such as 0.05, 0.15 and 0.25. The best model that was selected was a model with 2 hidden layers, with the neuron structure as:

- 1536 -> 256 -> 128 -> 64

with GELU activation function, a dropout of 0.25 for both the layers and trained for 8 epochs. The batch size used was 32, with a validation split of 0.25.

### Selecting Weights for Application of Genetic Algorithm

Once the model was trained, its weights were downloaded and copied into a final model structure. Histograms of the neural network weights were plotted. The weights of the final classifier layer were reduced to 4 bins. The centroids of these 4 bins were then selected as the 4 genes.

### Applying Genetic Algorithm

For the implementation of the Genetic Algorithm, the PyGAD library was used. The final layer of the neural network added at the end of the DINOv2 model had 128 * 64 = 8192 neurons. As such, taking the 4 genes from above, we created a population of size 6 generated randomly, where each member has 8192 genes. These weights then replace the neural network of the final model, and we run inference on the validation set only in order to select the 2 best children. The loss function is Triplet Loss, and we aim to minimize it. The fitness function is the inverse of the Triplet Loss. We evaluate only the validation set as we suffice with only having the 2 best-performing weight arrays for children as the parents for the next generation, which the validation set is enough to indicate; hence, there is no need to run the whole triplet loss model on the whole of the training data. After taking the top 2 members of the population as the parents for the next generation, 6 offspring are produced. This was done for 3 iterations, and the best offspring of the final iteration was selected as the weights array for the final model.

## Results

- The models were evaluated on criteria such as Accuracy, Confusion Matrices and Classification Reports, and DINOv2 was selected as the best model based on these criteria.
- A possible reason for DINOv2 outperforming ResNet-50 discussed in the Phase-1 report was considered to be the 'Attention' mechanism present in the DINOv2 model, which allowed it to have more detailed feature representations of the image.
- ResNet-50 is a CNN-based model, while DINOv2 is based on pre-trained visual models, trained with different Vision Transformers (ViTs). ViTs are significantly less biased towards local textures compared to CNNs. As such, CNNs, which have access to only a small receptive field, have to work with less information, while ViTs, thanks to attention layers, have a sufficiently sized receptive field to capture all the relevant information.
- DINOv2 was selected as the base model for the calculation of the category-wise percentage composition in the second phase.

## Discussion

The implementation of Genetic Algorithms and Contrastive Learning for classifying building usage types from Street-View images shows promising results. Several key points emerge from our work:

1. **Model Performance:** 
   - The use of DINOv2 over ResNet-50 significantly improved classification accuracy. DINOv2's attention mechanism allowed for better feature representation, making it more adept at distinguishing between different building types.
   - Our final model, which incorporated Genetic Algorithms to optimize weights, achieved notable performance improvements in classifying mixed-type buildings.

2. **Triplet Loss and Contrastive Learning:**
   - The application of Triplet Loss in training ensured that embeddings for images of the same class were closer together, while those of different classes were further apart. This method proved effective in handling the complex task of mixed-type building classification.
   - Contrastive Learning's ability to leverage unlabeled data was beneficial in enhancing the model's understanding of high-level features.

3. **Genetic Algorithm Optimization:**
   - The Genetic Algorithm successfully optimized the weights of the final classifier layer, improving model performance over several iterations.
   - This approach allowed us to fine-tune the model more precisely than standard backpropagation methods, highlighting the potential of evolutionary algorithms in deep learning.

4. **Challenges and Limitations:**
   - While the results are promising, the model's performance could still be improved by incorporating more diverse datasets and experimenting with different hyperparameters.
   - The complexity of mixed-type buildings in urban landscapes means that further research is needed to refine the model and address any edge cases or anomalies.

5. **Future Directions:**
   - Exploring additional methods such as advanced data augmentation techniques, incorporating more sophisticated Genetic Algorithms, and integrating other forms of self-supervised learning could further enhance model accuracy and robustness.
   - Applying the model to different urban environments and more diverse building types will test its generalizability and identify areas for improvement.

## Conclusion

- The combination of Genetic Algorithms and Contrastive Learning has shown promise in classifying mixed-type buildings from Street-View images. Further work could involve fine-tuning the model and exploring additional techniques to improve the accuracy and robustness of the classification process.

## Future Work

- Fine-tuning the model and exploring additional techniques to improve accuracy.
- Applying the model to a wider variety of mixed-type buildings and different urban settings.
- Integrating more advanced Genetic Algorithms and Contrastive Learning methods to enhance the model's performance.

## References

1. Jian Kang et al., "Building instance classification using street view images", ISPRS Journal of Photogrammetry and Remote Sensing 145 (2018).
2. Nitish Mutha, "Equirectangular toolbox", GitHub.
3. Maxime Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023.
4. Ross Wightman, Hugo Touvron, and Herve Jegou, "ResNet strikes back: An improved training procedure in timm", NeurIPS 2021.
5. Chen Sun, Abhinav Shrivastava, Saurabh Singh, Abhinav Gupta, "Revisiting Unreasonable Effectiveness of Data in Deep Learning Era", ICCV 2017.
6. Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", NeurIPS 2012.
7. Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton, "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020.
8. Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals, "Understanding deep learning requires rethinking generalization", ICLR 2017.
9. Yoav Goldberg, "Neural Network Methods for Natural Language Processing", Synthesis Lectures on Human Language Technologies, 2017.
10. Ian Goodfellow, Yoshua Bengio, Aaron Courville, "Deep Learning", MIT Press, 2016.
11. "PyGAD: Genetic Algorithm in Python", GitHub repository.

## Links

- [Dataset Folder](https://drive.google.com/file/d/1GwWhovPvTN_8R2vKHdSqnJUDXM0vxAxx/view?usp=drive_link)
