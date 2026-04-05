import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D
import keras

def getPrediction(filename):
    model1 = load_model('chest_xray.h5')
    image = load_img('static/images/'+filename, target_size=(256,256))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1,256,256,3)
    scores=model1.predict(img)
    confidence=np.max(scores)
    pred = np.argmax(scores).astype('int32')
    label=["COVID","NORMAL","PNEUMONIA"]

    last_conv_index=None
    for i,layer in enumerate(model1.layers):
        if isinstance(layer,Conv2D):
            last_conv_index = i
    
    inputs = tf.keras.Input(shape=(256,256,3))
    x = inputs

    # pass through all layers manually
    for layer in model1.layers:
        x = layer(x)

    model = tf.keras.Model(inputs, x)

    heatmap = make_gradcam_heatmap(img, model, last_conv_index)

    return label[pred],confidence,heatmap

def make_gradcam_heatmap(img_array, model, last_conv_layer_index, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions

    feature_extractor = tf.keras.Sequential(model.layers[:last_conv_layer_index + 1])
    classifier = tf.keras.Sequential(model.layers[last_conv_layer_index + 1:])
    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(img_array)
        tape.watch(conv_outputs)

        preds = classifier(conv_outputs)
        class_channel = tf.argmax(preds[0])
        loss = preds[:, class_channel]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(loss, conv_outputs)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = conv_outputs[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


