# Module-6_Fashion-MNIST-Classification

## Fashio MNIST Classification 
### For this assignment  i was tasked with classifying images using profile images to target marketing for different products. we are required to work with keras library in both Python and R for profile classification.
#### Python
### Challenges: Installing Keras proved to be challenging in the first 1 day as i was getting error notifications which upon further investigating i realized that i needed to install Tensorflow since keras work well with tensorflow. Unfortunately the code did not work within the same day so the next day on checking tensorflow was installed. HURRRRAY :)
#####  This is the code i used to install tensorflow:  

>pip install tensorflow --index-url=https://pypi.example.com/simple/
Looking in indexes: https://pypi.example.com/simple/
Requirement already satisfied: tensorflow in c:\users\doris\anaconda3\lib\site-packages (2.15.0)
Requirement already satisfied: tensorflow-intel==2.15.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow) (2.15.0)
Requirement already satisfied: absl-py>=1.0.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (2.1.0)
Requirement already satisfied: astunparse>=1.6.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.6.3)
Requirement already satisfied: flatbuffers>=23.5.26 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (23.5.26)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (0.5.4)
Requirement already satisfied: google-pasta>=0.1.1 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (0.2.0)
Requirement already satisfied: h5py>=2.9.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (3.9.0)
Requirement already satisfied: libclang>=13.0.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (16.0.6)
Requirement already satisfied: ml-dtypes~=0.2.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (0.2.0)
Requirement already satisfied: numpy<2.0.0,>=1.23.5 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.24.3)
Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (3.3.0)
Requirement already satisfied: packaging in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (23.1)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (4.25.2)
Requirement already satisfied: setuptools in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (68.0.0)
Requirement already satisfied: six>=1.12.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (2.4.0)
Requirement already satisfied: typing-extensions>=3.6.6 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (4.7.1)
Requirement already satisfied: wrapt<1.15,>=1.11.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.14.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (0.31.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.60.1)
Requirement already satisfied: tensorboard<2.16,>=2.15 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (2.15.2)
Requirement already satisfied: tensorflow-estimator<2.16,>=2.15.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (2.15.0)
Requirement already satisfied: keras<2.16,>=2.15.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (2.15.0)
Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\users\doris\anaconda3\lib\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.15.0->tensorflow) (0.38.4)
Requirement already satisfied: google-auth<3,>=1.6.3 in c:\users\doris\anaconda3\lib\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.27.0)
Requirement already satisfied: google-auth-oauthlib<2,>=0.5 in c:\users\doris\anaconda3\lib\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (1.2.0)
Requirement already satisfied: markdown>=2.6.8 in c:\users\doris\anaconda3\lib\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (3.4.1)
Requirement already satisfied: requests<3,>=2.21.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.31.0)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in c:\users\doris\anaconda3\lib\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.2.3)
Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\users\doris\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (5.3.2)
Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\doris\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (0.2.8)
Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\doris\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (4.9)
Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\doris\anaconda3\lib\site-packages (from google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (1.3.1)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\doris\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.0.4)
Requirement already satisfied: idna<4,>=2.5 in c:\users\doris\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\doris\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (1.26.16)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\doris\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2023.7.22)
Requirement already satisfied: MarkupSafe>=2.1.1 in c:\users\doris\anaconda3\lib\site-packages (from werkzeug>=1.0.1->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.1.1)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\users\doris\anaconda3\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (0.4.8)
Requirement already satisfied: oauthlib>=3.0.0 in c:\users\doris\anaconda3\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (3.2.2)
Note: you may need to restart the kernel to use updated packages.
## Upgrading pip
>pip install --upgrade pip
## Installing Keras and restarting the kernel
>pip install keras
Requirement already satisfied: keras in c:\users\doris\anaconda3\lib\site-packages (2.15.0)
Note: you may need to restart the kernel to use updated packages.
>pip install tensorflow
Requirement already satisfied: tensorflow in c:\users\doris\anaconda3\lib\site-packages (2.15.0)
Requirement already satisfied: tensorflow-intel==2.15.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow) (2.15.0)
Requirement already satisfied: absl-py>=1.0.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (2.1.0)
Requirement already satisfied: astunparse>=1.6.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.6.3)
Requirement already satisfied: flatbuffers>=23.5.26 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (23.5.26)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (0.5.4)
Requirement already satisfied: google-pasta>=0.1.1 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (0.2.0)
Requirement already satisfied: h5py>=2.9.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (3.9.0)
Requirement already satisfied: libclang>=13.0.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (16.0.6)
Requirement already satisfied: ml-dtypes~=0.2.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (0.2.0)
Requirement already satisfied: numpy<2.0.0,>=1.23.5 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.24.3)
Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (3.3.0)
Requirement already satisfied: packaging in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (23.1)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (4.25.2)
Requirement already satisfied: setuptools in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (68.0.0)
Requirement already satisfied: six>=1.12.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (2.4.0)
Requirement already satisfied: typing-extensions>=3.6.6 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (4.7.1)
Requirement already satisfied: wrapt<1.15,>=1.11.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.14.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (0.31.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (1.60.1)
Requirement already satisfied: tensorboard<2.16,>=2.15 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (2.15.2)
Requirement already satisfied: tensorflow-estimator<2.16,>=2.15.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (2.15.0)
Requirement already satisfied: keras<2.16,>=2.15.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorflow-intel==2.15.0->tensorflow) (2.15.0)
Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\users\doris\anaconda3\lib\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.15.0->tensorflow) (0.38.4)
Requirement already satisfied: google-auth<3,>=1.6.3 in c:\users\doris\anaconda3\lib\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.27.0)
Requirement already satisfied: google-auth-oauthlib<2,>=0.5 in c:\users\doris\anaconda3\lib\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (1.2.0)
Requirement already satisfied: markdown>=2.6.8 in c:\users\doris\anaconda3\lib\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (3.4.1)
Requirement already satisfied: requests<3,>=2.21.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.31.0)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\users\doris\anaconda3\lib\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in c:\users\doris\anaconda3\lib\site-packages (from tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.2.3)
Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\users\doris\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (5.3.2)
Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\doris\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (0.2.8)
Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\doris\anaconda3\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (4.9)
Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\doris\anaconda3\lib\site-packages (from google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (1.3.1)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\doris\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.0.4)
Requirement already satisfied: idna<4,>=2.5 in c:\users\doris\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (3.4)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\doris\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (1.26.16)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\doris\anaconda3\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2023.7.22)
Requirement already satisfied: MarkupSafe>=2.1.1 in c:\users\doris\anaconda3\lib\site-packages (from werkzeug>=1.0.1->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (2.1.1)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in c:\users\doris\anaconda3\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (0.4.8)
Requirement already satisfied: oauthlib>=3.0.0 in c:\users\doris\anaconda3\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<2,>=0.5->tensorboard<2.16,>=2.15->tensorflow-intel==2.15.0->tensorflow) (3.2.2)
Note: you may need to restart the kernel to use updated packages.
import keras
WARNING:tensorflow:From C:\Users\DORIS\anaconda3\Lib\site-packages\keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.
#### Loading the libraries from Keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelBinarizer
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Preprocess the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Defining the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
WARNING:tensorflow:From C:\Users\DORIS\anaconda3\Lib\site-packages\keras\src\backend.py:873: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.

WARNING:tensorflow:From C:\Users\DORIS\anaconda3\Lib\site-packages\keras\src\layers\pooling\max_pooling2d.py:161: The name tf.nn.max_pool is deprecated. Please use tf.nn.max_pool2d instead.

WARNING:tensorflow:From C:\Users\DORIS\anaconda3\Lib\site-packages\keras\src\optimizers\__init__.py:309: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.

Epoch 1/10
WARNING:tensorflow:From C:\Users\DORIS\anaconda3\Lib\site-packages\keras\src\utils\tf_utils.py:492: The name tf.ragged.RaggedTensorValue is deprecated. Please use tf.compat.v1.ragged.RaggedTensorValue instead.

WARNING:tensorflow:From C:\Users\DORIS\anaconda3\Lib\site-packages\keras\src\engine\base_layer_utils.py:384: The name tf.executing_eagerly_outside_functions is deprecated. Please use tf.compat.v1.executing_eagerly_outside_functions instead.

938/938 [==============================] - 25s 24ms/step - loss: 0.1823 - accuracy: 0.9436 - val_loss: 0.0502 - val_accuracy: 0.9843
Epoch 2/10
938/938 [==============================] - 22s 23ms/step - loss: 0.0487 - accuracy: 0.9848 - val_loss: 0.0438 - val_accuracy: 0.9845
Epoch 3/10
938/938 [==============================] - 22s 24ms/step - loss: 0.0341 - accuracy: 0.9893 - val_loss: 0.0374 - val_accuracy: 0.9876
Epoch 4/10
938/938 [==============================] - 22s 23ms/step - loss: 0.0277 - accuracy: 0.9911 - val_loss: 0.0301 - val_accuracy: 0.9907
Epoch 5/10
938/938 [==============================] - 22s 23ms/step - loss: 0.0223 - accuracy: 0.9926 - val_loss: 0.0262 - val_accuracy: 0.9909
Epoch 6/10
938/938 [==============================] - 22s 23ms/step - loss: 0.0175 - accuracy: 0.9944 - val_loss: 0.0251 - val_accuracy: 0.9922
Epoch 7/10
938/938 [==============================] - 22s 23ms/step - loss: 0.0146 - accuracy: 0.9952 - val_loss: 0.0250 - val_accuracy: 0.9913
Epoch 8/10
938/938 [==============================] - 22s 24ms/step - loss: 0.0130 - accuracy: 0.9958 - val_loss: 0.0251 - val_accuracy: 0.9923
Epoch 9/10
938/938 [==============================] - 22s 23ms/step - loss: 0.0102 - accuracy: 0.9968 - val_loss: 0.0272 - val_accuracy: 0.9925
Epoch 10/10
938/938 [==============================] - 22s 24ms/step - loss: 0.0095 - accuracy: 0.9970 - val_loss: 0.0311 - val_accuracy: 0.9904
313/313 [==============================] - 2s 7ms/step - loss: 0.0311 - accuracy: 0.9904
Test accuracy: 0.9904000163078308
import matplotlib.pyplot as plt

# Select two images from the test set
image1 = x_test[0]
image2 = x_test[1]

# Reshape the images for prediction
image1 = np.expand_dims(image1, axis=0)
image2 = np.expand_dims(image2, axis=0)

# Make predictions
prediction1 = np.argmax(model.predict(image1))
prediction2 = np.argmax(model.predict(image2))

# Print predictions
print("Prediction for image 1:", prediction1)
print("Prediction for image 2:", prediction2)

# Plot the images
plt.figure(figsize=(5, 5))
plt.subplot(121)
plt.imshow(np.squeeze(image1), cmap='gray')
plt.title(f"Prediction: {prediction1}")
plt.subplot(122)
plt.imshow(np.squeeze(image2), cmap='gray')
plt.title(f"Prediction: {prediction2}")
plt.show()
1/1 [==============================] - 0s 235ms/step
1/1 [==============================] - 0s 35ms/step
Prediction for image 1: 7
Prediction for image 2: 2

 ![image](https://github.com/Muenu/Module-6_Fashion-MNIST-Classification-/assets/115622275/47a0c9b5-e2a9-4d9e-a8ba-531573784361)

 #### R Code

 # I was unable to install the keras and tensorflow libraries into my R environment. My research(Github/Geek2 Geek/Chatgpt) on various ways to resolve the issue were met with different errors as you can see below. Will try tomorrow to see if the code will work maybe i need to do more research in installing the code.

> library(keras)
> library(reticulate)
> library(tensorflow)
> install.packages("reticulate")
Error in install.packages : Updating loaded packages

Restarting R session...

> install.packages("reticulate")
WARNING: Rtools is required to build R packages but is not currently installed. Please download and install the appropriate version of Rtools before proceeding:

https://cran.rstudio.com/bin/windows/Rtools/
Installing package into ‘C:/Users/DORIS/AppData/Local/R/win-library/4.3’
(as ‘lib’ is unspecified)
trying URL 'https://cran.rstudio.com/bin/windows/contrib/4.3/reticulate_1.35.0.zip'
Content type 'application/zip' length 2180048 bytes (2.1 MB)
downloaded 2.1 MB

package ‘reticulate’ successfully unpacked and MD5 sums checked

The downloaded binary packages are in
	C:\Users\DORIS\AppData\Local\Temp\RtmpIT76eo\downloaded_packages
> library(reticulate)
> install_tensorflow()
Error in install_tensorflow() : 
  could not find function "install_tensorflow"
> keras::install_keras().
Error: unexpected symbol in "keras::install_keras()."
> keras::install_keras()
Using Python: C:/Users/DORIS/AppData/Local/Programs/Python/Python312/python.exe
Creating virtual environment "r-tensorflow" ... 
+ "C:/Users/DORIS/AppData/Local/Programs/Python/Python312/python.exe" -m venv "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow"
Done!
Installing packages: pip, wheel, setuptools
+ "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe" -m pip install --upgrade pip wheel setuptools
Requirement already satisfied: pip in c:\users\doris\docume~1\virtua~1\r-tens~1\lib\site-packages (23.2.1)
Collecting pip
  Obtaining dependency information for pip from https://files.pythonhosted.org/packages/8a/6a/19e9fe04fca059ccf770861c7d5721ab4c2aebc539889e97c7977528a53b/pip-24.0-py3-none-any.whl.metadata
  Using cached pip-24.0-py3-none-any.whl.metadata (3.6 kB)
Collecting wheel
  Obtaining dependency information for wheel from https://files.pythonhosted.org/packages/c7/c3/55076fc728723ef927521abaa1955213d094933dc36d4a2008d5101e1af5/wheel-0.42.0-py3-none-any.whl.metadata
  Using cached wheel-0.42.0-py3-none-any.whl.metadata (2.2 kB)
Collecting setuptools
  Obtaining dependency information for setuptools from https://files.pythonhosted.org/packages/bb/0a/203797141ec9727344c7649f6d5f6cf71b89a6c28f8f55d4f18de7a1d352/setuptools-69.1.0-py3-none-any.whl.metadata
  Using cached setuptools-69.1.0-py3-none-any.whl.metadata (6.1 kB)
Using cached pip-24.0-py3-none-any.whl (2.1 MB)
Using cached wheel-0.42.0-py3-none-any.whl (65 kB)
Using cached setuptools-69.1.0-py3-none-any.whl (819 kB)
Installing collected packages: wheel, setuptools, pip
  Attempting uninstall: pip
    Found existing installation: pip 23.2.1
    Uninstalling pip-23.2.1:
      Successfully uninstalled pip-23.2.1
Successfully installed pip-24.0 setuptools-69.1.0 wheel-0.42.0
Virtual environment 'r-tensorflow' successfully created.
Using virtual environment "r-tensorflow" ...
+ "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe" -m pip install --upgrade --no-user "tensorflow==2.13.*" tensorflow-hub tensorflow-datasets scipy requests Pillow h5py pandas pydot
ERROR: Could not find a version that satisfies the requirement tensorflow==2.13.* (from versions: none)
ERROR: No matching distribution found for tensorflow==2.13.*
Error: Error installing package(s): "\"tensorflow==2.13.*\"", "tensorflow-hub", "tensorflow-datasets", "scipy", "requests", "Pillow", "h5py", "pandas", "pydot"
> library(keras)
> 
> 
> mnist <- dataset_mnist()
Error: Valid installation of TensorFlow not found.

Python environments searched for 'tensorflow' package:
 C:\Users\DORIS\Documents\.virtualenvs\r-tensorflow\Scripts\python.exe

Python exception encountered:
 Traceback (most recent call last):
  File "C:\Users\DORIS\AppData\Local\R\win-library\4.3\reticulate\python\rpytools\loader.py", line 119, in _find_and_load_hook
    return _run_hook(name, _hook)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\DORIS\AppData\Local\R\win-library\4.3\reticulate\python\rpytools\loader.py", line 93, in _run_hook
    module = hook()
             ^^^^^^
  File "C:\Users\DORIS\AppData\Local\R\win-library\4.3\reticulate\python\rpytools\loader.py", line 117, in _hook
    return _find_and_load(name, import_)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ModuleNotFoundError: No module named 'tensorflow'


You can install TensorFlow using the install_tensorflow() function.
> install_tensorflow()
Error in install_tensorflow() : 
  could not find function "install_tensorflow"
> library(reticulate)
> py_install("tensorflow==2.7.0")
Using virtual environment "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow" ...
+ "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe" -m pip install --upgrade --no-user "tensorflow==2.7.0"
ERROR: Could not find a version that satisfies the requirement tensorflow==2.7.0 (from versions: none)
ERROR: No matching distribution found for tensorflow==2.7.0
Error: Error installing package(s): "\"tensorflow==2.7.0\""
> py_install("tensorflow==2.7.0")
Using virtual environment "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow" ...
+ "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe" -m pip install --upgrade --no-user "tensorflow==2.7.0"
ERROR: Could not find a version that satisfies the requirement tensorflow==2.7.0 (from versions: none)
ERROR: No matching distribution found for tensorflow==2.7.0
Error: Error installing package(s): "\"tensorflow==2.7.0\""
> library(reticulate)
> py_config()
python:         C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe
libpython:      C:/Users/DORIS/AppData/Local/Programs/Python/Python312/python312.dll
pythonhome:     C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow
version:        3.12.0 (tags/v3.12.0:0fb18b0, Oct  2 2023, 13:03:39) [MSC v.1935 64 bit (AMD64)]
Architecture:   64bit
numpy:           [NOT FOUND]
tensorflow:     [NOT FOUND]

NOTE: Python version was forced by import("tensorflow")
> py_install("pip")
Using virtual environment "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow" ...
+ "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe" -m pip install --upgrade --no-user pip
Requirement already satisfied: pip in c:\users\doris\docume~1\virtua~1\r-tens~1\lib\site-packages (24.0)
> py_install("setuptools")
Using virtual environment "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow" ...
+ "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe" -m pip install --upgrade --no-user setuptools
Requirement already satisfied: setuptools in c:\users\doris\docume~1\virtua~1\r-tens~1\lib\site-packages (69.1.0)
> py_install("tensorflow==2.7.0")
Using virtual environment "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow" ...
+ "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe" -m pip install --upgrade --no-user "tensorflow==2.7.0"
ERROR: Could not find a version that satisfies the requirement tensorflow==2.7.0 (from versions: none)
ERROR: No matching distribution found for tensorflow==2.7.0
Error: Error installing package(s): "\"tensorflow==2.7.0\""
> py_install("tensorflow==2.6.0")
Using virtual environment "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow" ...
+ "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe" -m pip install --upgrade --no-user "tensorflow==2.6.0"
ERROR: Could not find a version that satisfies the requirement tensorflow==2.6.0 (from versions: none)
ERROR: No matching distribution found for tensorflow==2.6.0
Error: Error installing package(s): "\"tensorflow==2.6.0\""
> mnist <- dataset_mnist()
List of 22
 $ python              : chr "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe"
 $ libpython           : chr "C:/Users/DORIS/AppData/Local/Programs/Python/Python312/python312.dll"
 $ pythonhome          : chr "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow"
 $ pythonpath          : chr "C:\\Users\\DORIS\\AppData\\Local\\R\\win-library\\4.3\\reticulate\\config;C:\\Users\\DORIS\\AppData\\Local\\Pro"| __truncated__
 $ prefix              : chr "C:\\Users\\DORIS\\DOCUME~1\\VIRTUA~1\\R-TENS~1"
 $ exec_prefix         : chr "C:\\Users\\DORIS\\DOCUME~1\\VIRTUA~1\\R-TENS~1"
 $ base_exec_prefix    : chr "C:\\Users\\DORIS\\AppData\\Local\\Programs\\Python\\PYTHON~1"
 $ virtualenv          : chr "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow"
 $ virtualenv_activate : chr ""
 $ executable          : chr "C:\\Users\\DORIS\\DOCUME~1\\VIRTUA~1\\R-TENS~1\\Scripts\\python.exe"
 $ base_executable     : chr "C:\\Users\\DORIS\\AppData\\Local\\Programs\\Python\\PYTHON~1\\python.exe"
 $ version_string      : chr "3.12.0 (tags/v3.12.0:0fb18b0, Oct  2 2023, 13:03:39) [MSC v.1935 64 bit (AMD64)]"
 $ version             : chr "3.12"
 $ architecture        : chr "64bit"
 $ anaconda            : logi FALSE
 $ conda               : chr "False"
 $ numpy               : NULL
 $ required_module     : chr "tensorflow"
 $ required_module_path: NULL
 $ available           : logi TRUE
 $ python_versions     : chr "C:/Users/DORIS/Documents/.virtualenvs/r-tensorflow/Scripts/python.exe"
 $ forced              : chr "import(\"tensorflow\")"
 - attr(*, "class")= chr "py_config"
Error: Python module tensorflow.keras was not found.

Detected Python configuration:


> library(keras)
> library(R.matlab)
Error in library(R.matlab) : there is no package called ‘R.matlab’
>
> ## R code- Fashion MNIST library  
# I had already written the code following the guidance of the the course material.
# Loading data set 

mnist <- dataset_mnist()
x_train <-mnist$train$x
y_train <-mnist$train$y
x_test <-mnist$test$x
y_test <-mnist$test$y


# Process the data 
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
x_train <- x_train / 255
x_test <- x_test / 255
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Create a sequential model 
model<-keras_model_Sequential()

#Adding layers to the  model
model%>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')


# Compile the model
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
