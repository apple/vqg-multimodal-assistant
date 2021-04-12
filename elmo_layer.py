#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#
"""
This file contains Elmo Embedding Layer class
"""
import os
import tensorflow as tf
import tensorflow_hub as hub

os.environ['TFHUB_CACHE_DIR'] = os.path.join(os.getcwd(), 'misc')

# There are some import statement problems with the commands below, so I'm commenting it out.
# from tensorflow.keras import backend as K
# from keras.engine import Layer
# from tensorflow.keras.layers import Layer


# class ElmoEmbeddingLayer(Layer):
"""
This layer works with our keras model. There are some import statement problems so I'm commenting it out.
We fell back to using the module ELMo Embedding below.
"""
#     def __init__(self,
#                  pooling="first",
#                  trainable=True,
#                  **kwargs):
#         self.dimensions = 1024
#         self.trainable = trainable
#         self.pooling = pooling
#         super(ElmoEmbeddingLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
#                                name="{}_module".format(self.name))
#
#         self.trainable_weights += tf.trainable_variables(scope="^{}_module/.*".format(self.name))
#         super(ElmoEmbeddingLayer, self).build(input_shape)
#
#     def call(self, x, mask=None):
#         if self.pooling == 'first':
#             result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
#                       as_dict=True,
#                       signature='elmo',
#                       )['elmo']
#         elif self.pooling == 'mean':
#             result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
#                       as_dict=True,
#                       signature='default',
#                       )['default']
#         return result
#
#     def compute_mask(self, inputs, mask=None):
#         return K.not_equal(inputs, '<PAD>')
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], self.dimensions


def ELMoEmbedding(inp):
    """
    This module is used in a Lambda layer with our keras model. The model is not performing well. One hypothesis is lambda layers don't train.
    Replacing with above Class works but there are some import statement issues.
    :param inp:
    :return:
    """
    trainable = True
    pooling = "first"
    x = inp
    print('Instantiating hub layer')
    embed = hub.Module('https://tfhub.dev/google/elmo/2', trainable=trainable)
    print('Loaded hub layer')
    if pooling == 'mean':
        return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
    else:
        return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["word_emb"]