#
# This code is based on the paper A Neural Algorithm of Artistic Style by
# Gatys et al. 2015. It is an adapted version of the blog post discussing
# the paper https://harishnarayanan.org/writing/artistic-style-transfer/.
#

from pprint import pprint

import time
from PIL import Image
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b

# Set the size of the images to process.
width = 512 #773 #512
height = 512

# Set the weight for content and style loss. Variation weight is for smoothing
# final output image
content_weight = 0.025
style_weight = 5
total_variation_weight = 2.0 #1

# Load the content image, resize it
content_image_path = 'data/elephant.jpg'
content_image = Image.open(content_image_path)
content_image = content_image.resize((width, height))
#content_image.show()
# Load the style image and resize it
style_image_path = 'data/style/block.jpg'
style_image = Image.open(style_image_path)
style_image = style_image.resize((width, height))
#style_image.show()

# Add an extra dimension to content and style images to match the dimensions
# used by the backend i.e. [1, 512, 512, 3]
content_array = np.asarray(content_image, dtype='float32')
content_array = np.expand_dims(content_array, axis=0)
print(content_array.shape)

style_array = np.asarray(style_image, dtype='float32')
style_array = np.expand_dims(style_array, axis=0)
print(style_array.shape)


# We are going to use VGG16 pretrained on ImageNet. In order to match the VGG16
# paper Very Deep Convolutional Networks for Large-Scale Image Recognition by
# Simomyan and Zisserman 2015 we need to subtract the mean RGB values from all
# channels. Those values have been computed on the ImageNet dataset. We also need
# to flip the ordering of the channels to BGR.
content_array[:, :, :, 0] -= 103.939
content_array[:, :, :, 1] -= 116.779
content_array[:, :, :, 2] -= 123.68
style_array[:, :, :, 0] -= 103.939
style_array[:, :, :, 1] -= 116.779
style_array[:, :, :, 2] -= 123.68
content_array = content_array[:, :, :, ::-1]
style_array = style_array[:, :, :, ::-1]

# Create the backend variables. In our case tensorflow.
content_image = K.variable(content_array)
style_image = K.variable(style_array)
combination_image = K.placeholder((1, height, width, 3))

# Concatenate all tensors

input_tensor = K.concatenate([content_image,
                                    style_image,
                                    combination_image], axis=0)

# Load the VGG16 model from Keras. We are only interested in getting the features
# from the different layers hence we omit the dense layers at the top.
model = VGG16(input_tensor=input_tensor, weights='imagenet',
              include_top=False)

# Store layers of the model. We'll need that to refer to the layers we want to
# use for the transfer.
layers = dict([(layer.name, layer.output) for layer in model.layers])
#pprint(layers)

# Define the total loss. We'll add to this in stages
loss = K.variable(0.)

# Define the content loss. This is the distance between the feature representation
# and the combination image. Here we use block2_conv2 to draw content features.
def content_loss(content, combination):
    return K.sum(K.square(combination - content))

layer_features = layers['block2_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss += content_weight * content_loss(content_image_features,
                                      combination_features)

# Define the style loss. In order to separate is completely from the content
# this needs to be represented as a Gram matrix. Gram matrix is used to capture
# linear independence. Its terms are proportional to the covariance and captures
# information about features that tend to activate together. The Gram matrix can
# be computed efficiently by reshaping the feature spaces and taking an outer product.
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# The style loss is the Frobenius norm of the style image and the combination.
# Which layers to use for style transfer can be changed to your taste.
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']
for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

# Using only these losses results in noisy output. Add a regularization term
# in the form of total variation loss for spatial smootheness.
def total_variation_loss(x):
    a = K.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = K.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

loss += total_variation_weight * total_variation_loss(combination_image)

# Define the gradients of the combination image w.r.t. the loss function
grads = K.gradients(loss, combination_image)

# Class that calculates loss and gradients on one pass. We do this becayse
# optimizer requires two separate functions for those but that is inefficient.
outputs = [loss]
outputs += grads
f_outputs = K.function([combination_image], outputs)
#print(f_outputs)
def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

iterations = 10

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

x = x.reshape((height, width, 3))
x = x[:, :, ::-1]
x[:, :, 0] += 103.939
x[:, :, 1] += 116.779
x[:, :, 2] += 123.68
x = np.clip(x, 0, 255).astype('uint8')
print("Image generated from {0} and {1}".format(content_image_path, style_image_path))
img = Image.fromarray(x)
img.save("out.jpg")
