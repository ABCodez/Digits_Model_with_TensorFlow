import tensorflow as tf
import matplotlib.pyplot as plt

# CREATING A NEURAL MODEL THAT CAN CLASSIFY NUMBERS

# get data then train and test the datasets
# first we train the neural network img with its appropriate label, then test it with testImg and labels
(trainImg, trainLabel), (testImg, testLabel) = tf.keras.datasets.mnist.load_data()

# scale the images down in order to let our neural network easily process the images (0 - 255px -> 0-1px)
trainImg = trainImg / 255.0
testImg = testImg / 255.0

# visualize the data
print(trainImg.shape)
print(testImg.shape)
print(trainLabel)

# display the first image using plt
plt.imshow(trainImg[0], cmap='gray')
plt.show()

# define neural network model
neuralModel = tf.keras.models.Sequential()  # Use sequential type modelling
# This will flatten our img to single lines and feed them to our neural network
neuralModel.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
neuralModel.add(
    tf.keras.layers.Dense(128, activation='relu'))  # usually multiple of 8, relu most widely used activation
neuralModel.add(tf.keras.layers.Dense(10, activation='softmax'))  # we have num from 0-9 therefore 10

# compile and create our model
neuralModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])  # since we use a multiClass Classification we use loss ...

# train the model
neuralModel.fit(trainImg, trainLabel,
                epochs=3)  # train 3x, dont do too high number of model will start to OVERFIT. overfit (the model will try to memorize certain parts of the images, which isnt useful for generalziation)

# check model accuracy on the test data now
lossVal, accVal = neuralModel.evaluate(testImg, testLabel)
print("Test Accuracy: ", accVal)

# Save model so we can use it for the future
neuralModel.save('ABCodez_Digits_Model')
