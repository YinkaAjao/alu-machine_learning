#!/usr/bin/env python3
"""Neural Style Transfer - Task 8: Compute Gradients"""
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


class NST:
    """Performs tasks for neural style transfer."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize NST instance."""
        if (not isinstance(style_image, np.ndarray)
                or style_image.ndim != 3
                or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if (not isinstance(content_image, np.ndarray)
                or content_image.ndim != 3
                or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales image so pixels are in [0, 1] and max side is 512 px."""
        if (not isinstance(image, np.ndarray)
                or image.ndim != 3
                or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        h, w = image.shape[0], image.shape[1]
        if h > w:
            new_h, new_w = 512, int(w * 512 / h)
        else:
            new_h, new_w = int(h * 512 / w), 512
        image = tf.constant(image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bicubic(image, [new_h, new_w])
        image = image / 255.0
        return tf.clip_by_value(image, 0.0, 1.0)

    def load_model(self):
        """Creates the Keras model used to calculate cost."""
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')
        vgg.trainable = False

        config = vgg.get_config()
        for layer_conf in config['layers']:
            if layer_conf['class_name'] == 'MaxPooling2D':
                layer_conf['class_name'] = 'AveragePooling2D'

        custom_vgg = tf.keras.Model.from_config(config)
        custom_vgg.set_weights(vgg.get_weights())
        custom_vgg.trainable = False

        target_layers = self.style_layers + [self.content_layer]
        model_outputs = [custom_vgg.get_layer(name).output
                         for name in target_layers]

        self.model = tf.keras.Model(inputs=custom_vgg.input,
                                    outputs=model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of a layer output."""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        shape = tf.shape(input_layer)
        h, w = shape[1], shape[2]

        gram = tf.einsum('bijc,bijd->bcd', input_layer, input_layer)
        return gram / tf.cast(h * w, tf.float32)

    def generate_features(self):
        """Extracts style and content features from the model."""
        vgg19 = tf.keras.applications.vgg19

        style_prep = vgg19.preprocess_input(self.style_image * 255)
        style_outputs = self.model(style_prep)
        self.gram_style_features = [self.gram_matrix(out)
                                    for out in style_outputs[:-1]]

        content_prep = vgg19.preprocess_input(self.content_image * 255)
        content_outputs = self.model(content_prep)
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Calculates style cost for a single layer."""
        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[3]
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != (1, c, c)):
            raise TypeError(
                "gram_target must be a tensor of shape [1, {}, {}]".format(
                    c, c))

        gram_style = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """Calculates the total style cost for the generated image."""
        n = len(self.style_layers)
        if not isinstance(style_outputs, list) or len(style_outputs) != n:
            raise TypeError(
                "style_outputs must be a list with a length of {}".format(n))

        costs = [self.layer_style_cost(style_outputs[i],
                                       self.gram_style_features[i])
                 for i in range(n)]
        return tf.add_n(costs) / n

    def content_cost(self, content_output):
        """Calculates the content cost for the generated image."""
        s = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable))
                or content_output.shape != s):
            raise TypeError(
                "content_output must be a tensor of shape {}".format(s))

        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image."""
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != s):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s))

        vgg19 = tf.keras.applications.vgg19
        preprocessed = vgg19.preprocess_input(generated_image * 255)
        outputs = self.model(preprocessed)

        J_style = self.style_cost(list(outputs[:-1]))
        J_content = self.content_cost(outputs[-1])
        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """Calculates the gradients for the generated image.

        Args:
            generated_image: tf.Tensor or tf.Variable shape (1, nh, nw, 3)

        Returns:
            (gradients, J_total, J_content, J_style)
        """
        s = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != s):
            raise TypeError(
                "generated_image must be a tensor of shape {}".format(s))

        with tf.GradientTape() as tape:
            J, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J, generated_image)
        return gradients, J, J_content, J_style
