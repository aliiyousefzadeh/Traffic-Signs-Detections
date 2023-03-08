## Traffic Signs Classification using Deep Learning

# Inrtoduction
This project focuses on classifying traffic signs into their respective categories using deep learning. In this project, a convolutional neural network (CNN) is used for image classification. The dataset used is the German Traffic Sign Recognition Benchmark (GTSRB), which consists of 43 different classes of traffic signs with images of varying sizes. The dataset is divided into three sets: training, validation, and testing.

# Technical Description
The project is implemented using Python 3 and Keras with TensorFlow backend. First, the dataset is loaded using the pickle library. The loaded data is then split into training, validation, and test sets. After loading the data, the dimensions of the images are checked to make sure they are 32 x 32 x 3, which is the required input shape for the CNN. Then, the data is preprocessed by converting the images to grayscale, applying histogram equalization to improve contrast, and normalizing the pixel values to be between 0 and 1.

A CNN model is then defined using the Keras Sequential API. The model consists of two convolutional layers, followed by a max-pooling layer, two more convolutional layers, another max-pooling layer, a flattening layer, and two fully connected (Dense) layers. The activation function used is Rectified Linear Unit (ReLU), and dropout is applied to reduce overfitting.

After defining the model, it is compiled using the Adam optimizer and categorical cross-entropy loss function. A learning rate scheduler is also used to decrease the learning rate over time. A ModelCheckpoint is used to save the best model during training.

Next, data augmentation is applied using Keras ImageDataGenerator to generate new images by performing various transformations on the original images, such as rotation, zooming, and shifting. The augmented data is then used to train the CNN.

Finally, the trained model is evaluated on the test set, and the accuracy is reported.

# Results
The trained model achieved an accuracy of 97.68% on the test set. The distribution of the training dataset is also plotted to show that the dataset is balanced.

# Acknowledgment
The dataset used in this project is the German Traffic Sign Recognition Benchmark (GTSRB), which can be found at https://bitbucket.org/jadslim/german-traffic-signs/src/master/. The project was completed as part of the Udemy Self-Driving Car program.

# Future Improvements
Future improvements could include experimenting with different CNN architectures, such as deeper or wider networks, to improve accuracy. Other techniques, such as transfer learning, could also be explored. Additionally, the dataset could be expanded by collecting more images or using data augmentation techniques that generate more diverse images.
