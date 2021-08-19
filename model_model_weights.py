import tensorflow as tf
from tensorflow import keras
import pycuda.autoinit


def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE
    # path = f"{getcwd()}/../tmp2/mnist.npz"
 



    model = tf.keras.Sequential([

        keras.layers.Conv2D(filters=20, kernel_size=(5, 5), input_shape=(28, 28, 1), activation="relu", strides=(1,1), padding="valid",name="conv1"),

        keras.layers.MaxPooling2D(2, 2),

        keras.layers.Conv2D(filters=50, activation="relu", kernel_size=(5, 5), strides=(1,1),padding="valid",name="conv2"),

        keras.layers.MaxPooling2D(2, 2),
        
        keras.layers.Flatten(),
        
        keras.layers.Dense(500, activation="relu",name="fc1"),

        keras.layers.Dense(10, activation="sigmoid",name="fc2")
#         keras.layers.Dense(10,name="fc2"),
        
#         keras.layers.Activation("sigmoid")

    ])

    print(model.summary())
    
    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    # YOUR CODE STARTS HERE
    #


    training_images=training_images/255
    test_images=test_images/255

    training_images = training_images.reshape(60000, 28, 28, 1)
    #
    test_images = test_images.reshape(10000, 28, 28, 1)
    # YOUR CODE ENDS HERE
      
        
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    # model fitting
    history = model.fit(training_images,
                        training_labels,
                        epochs=10,
                        batch_size=1000,
                        validation_data=(test_images, test_labels))

    return model
#     return model,model.get_weights()

# model,get_weights=train_mnist_conv()