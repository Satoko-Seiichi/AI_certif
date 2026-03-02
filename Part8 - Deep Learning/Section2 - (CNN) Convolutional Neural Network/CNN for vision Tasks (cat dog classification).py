import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
######Part1 - Data Preprocessing
"""Preprocessing the Training set""" 
train_datagen = ImageDataGenerator(rescale = 1./255, #feature scaling
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)#to prevent over feading
#importing the training set
training_set = train_datagen.flow_from_directory(r"C:\Users\youne\Desktop\AI_certif\Part8 - Deep Learning\Data\dataset\training_set",
                                                 target_size = (64, 64),#the size of the images that are going to be feeded into the neural network
                                                 batch_size = 32,
                                                 class_mode = 'binary'#only Cat Or dog
                                                 )
"""Preprocessing the Test set"""
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(r'C:\Users\youne\Desktop\AI_certif\Part8 - Deep Learning\Data\dataset\test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
######Part2 - Building the CNN
"""initialising the CNN"""
cnn = tf.keras.models.Sequential()
"""Step1 - Convolution"""
cnn.add(tf.keras.layers.Conv2D(filters=32, 
                               kernel_size = 3, #3x3 filter size or the feature extractor 
                               activation = 'relu',
                               input_shape = [64, 64, 3]
                               ))#convolutional neural network Layer architecture
"""Step2 - Pooling"""
cnn.add(tf.keras.layers.MaxPool2D(pool_size =2, strides=2#shifting the pixel by 2 cells
    ))
"""Adding a Second convolutional layer"""
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size = 3, activation = 'relu'))# you multiply the feature matrix by the image matrix(without input shape)
cnn.add(tf.keras.layers.MaxPool2D(pool_size =2, strides=2))#you do the max pooling or taking the highest pixels value form the feature matrix
    
"""Step3 - Flattening"""
cnn.add(tf.keras.layers.Flatten())
"""Step3 - Full connection"""
cnn.add(tf.keras.layers.Dense(units = 128, activation ='relu'))
"""Step3 - Output Layer"""
cnn.add(tf.keras.layers.Dense(units = 1, activation ='sigmoid'))

######Part3 - Training the CNN (Making the Brain Smart)
"""Compiling the CNN (Optimizing the results by the loss function)"""
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])
"""Training the CNN on the Training and evaluating it on the Test set at the same time"""
cnn.fit(x = training_set, validation_data = test_set,
        epochs = 2#how many time to play it and adjust it
        )
######Part4 - Making a Single Prediction
import numpu as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img(r"C:\Users\youne\Desktop\AI_certif\Part8 - Deep Learning\Data\dataset\Single_prediction", target_size = (64, 64))
test_image = image.img_to_array(test_image) #the predict methode needs a 2d Arrays
test_image = np.expand_dims(test_image, axis = 0) #the dimension of the batch of the image 
result = cnn.predict(test_image)
training_set.class_indices #cat correspond to 0 and dog correspond to 1

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
