import tensorflow as tf
import numpy as np
import copy
from keras import layers, regularizers
from keras.optimizers import SGD
from keras.utils.generic_utils import get_custom_objects
from keras.layers.core import Activation
import keras.backend as K


class MyModel(tf.Module):

    def __init__(self, type, num_switches, normalizer, data_shape_lst, regularizer_ODE=0, regularizer_classif=0,
                 use_bias=False, **kwargs):
        """
        Initialize regression model of specified type with relevant parameters
        :param type: str naming type of model
        :param num_switches: int number of allowed models to switch between
        :param normalizer: tf.normalizer to use for data normalization
        :param data_shape_lst: array of two numbers - number of entries in input data and number of allowed coeffs
                               for each model
        :param regularizer: tf.regularizer to use for regression regularization
        :param kwargs:
        """

        data_shape = data_shape_lst[0]
        take_coeffs = data_shape_lst[1]

        super().__init__(**kwargs)

        # initializer = tf.keras.initializers.Constant(value=[-1, 0, 0, 0, 1, 0, 0, 0, 0, -1])

        if type == 'softmax_switch_sparse':
            inputs = tf.keras.Input(shape=data_shape, name="dataset")
            inputs_norm = normalizer(inputs[:, :take_coeffs])
            inputs_norm = tf.concat([inputs[:, 0:1], inputs_norm], 1)
            switch = layers.Dense(num_switches, use_bias=use_bias, activation='softmax')(inputs_norm)

            if regularizer_ODE == 0:
                x = layers.Dense(num_switches, use_bias=use_bias)(inputs_norm)
            else:
                x = layers.Dense(num_switches, use_bias=use_bias, kernel_regularizer=regularizer_ODE)(inputs_norm)

            # x = layers.Dense(1, use_bias=False, kernel_regularizer=regularizer)(inputs_norm)
            # for i in range(num_switches-1):
            #     x = tf.concat([x, layers.Dense(1, use_bias=False, kernel_regularizer=regularizer)(inputs_norm)], axis=1)

            outputs = tf.keras.layers.Dot(axes=1)([x, switch])

            self.model = tf.keras.Model(inputs, outputs, name="softmax_switch_model")

        elif type == 'softmax_switch_double_sparse':
            inputs = tf.keras.Input(shape=data_shape, name="dataset")
            inputs_norm = normalizer(inputs[:, :take_coeffs])
            # inputs_norm = tf.concat([inputs[:, 0:1], inputs_norm], 1)
            # inputs_norm[:, 1:take_coeffs] = normalizer(inputs[:, 1:take_coeffs])  # if only part of coeffs are desired

            if regularizer_classif == 0:
                switch = layers.Dense(num_switches, use_bias=use_bias, activation='softmax')(inputs_norm)
            else:
                switch = layers.Dense(num_switches, use_bias=use_bias, kernel_regularizer=regularizer_classif,
                                      activation='softmax')(inputs_norm)
            if regularizer_ODE == 0:
                x = layers.Dense(num_switches, use_bias=use_bias)(inputs_norm)
            else:
                x = layers.Dense(num_switches, use_bias=use_bias, kernel_regularizer=regularizer_ODE)(inputs_norm)

            # for i in range(num_switches-1):
            #     x = tf.concat([x, layers.Dense(1, use_bias=False, kernel_regularizer=regularizer)(inputs_norm)], axis=1)

            outputs = tf.keras.layers.Dot(axes=1)([switch, x])
            # outputs = tf.keras.layers.Dot(axes=1)([switch[:, 0:x.shape[1]], x])

            self.model = tf.keras.Model(inputs, outputs, name="softmax_switch_model")

        elif type == 'mysoftmax_switch_double_sparse':
            # add mysoftmax to keras
            def mysoftmax(x, beta_softmax=2.0):
                return K.softmax(beta_softmax * x)

            get_custom_objects().update({'mysoftmax': Activation(mysoftmax)})

            inputs = tf.keras.Input(shape=data_shape, name="dataset")
            inputs_norm = normalizer(inputs[:, :take_coeffs])
            # inputs_norm = tf.concat([inputs[:, 0:1], inputs_norm], 1)
            # inputs_norm[:, 1:take_coeffs] = normalizer(inputs[:, 1:take_coeffs])  # if only part of coeffs are desired

            if regularizer_classif == 0:
                switch = layers.Dense(num_switches, use_bias=use_bias, activation='mysoftmax')(inputs_norm)
            else:
                switch = layers.Dense(num_switches, use_bias=use_bias, kernel_regularizer=regularizer_classif,
                                      activation='mysoftmax')(inputs_norm)
            if regularizer_ODE == 0:
                x = layers.Dense(num_switches, use_bias=use_bias)(inputs_norm)
            else:
                x = layers.Dense(num_switches, use_bias=use_bias, kernel_regularizer=regularizer_ODE)(inputs_norm)

            # for i in range(num_switches-1):
            #     x = tf.concat([x, layers.Dense(1, use_bias=False, kernel_regularizer=regularizer)(inputs_norm)], axis=1)

            outputs = tf.keras.layers.Dot(axes=1)([switch, x])
            # outputs = tf.keras.layers.Dot(axes=1)([switch[:, 0:x.shape[1]], x])

            self.model = tf.keras.Model(inputs, outputs, name="my_softmax_switch_doubleSparse_model")

        elif type == 'softmax_switch_2state':
            inputs = tf.keras.Input(shape=data_shape, name="dataset")
            inputs_norm = normalizer(inputs[:, 1:take_coeffs])
            inputs_norm = tf.concat([inputs[:, 0:1], inputs_norm], 1)
            switch = layers.Dense(2, use_bias=use_bias, activation='softmax')(inputs_norm)

            # x1 = layers.Dense(1, use_bias=False, kernel_regularizer=regularizer)(inputs_norm)
            # x2 = layers.Dense(1, use_bias=False, kernel_regularizer=regularizer)(inputs_norm)
            # x = layers.Concatenate(axis=1)([x1, x2])
            x = layers.Dense(2, use_bias=use_bias, kernel_regularizer=regularizer_ODE)(inputs_norm)

            outputs = tf.keras.layers.Dot(axes=1)([x, switch])

            self.model = tf.keras.Model(inputs, outputs, name="softmax_switch_2state_model")

        elif type == 'LR':  # type is Fully connected Dense LR
            # self.model = tf.keras.Sequential([normalizer, layers.Dense(units=1, use_bias=False,
            #                                                            kernel_initializer=initializer)])
            # self.model = tf.keras.Sequential([normalizer, layers.Dense(units=1, use_bias=False)])
            self.model = tf.keras.Sequential([normalizer, layers.Dense(units=1, use_bias=use_bias)])

        self.model.summary()

    def model_compile(self, optimizer):
        """
        Compiler of the model
        :param optimizer: tf.optimizer to use
        :return: compiled model
        """
        self.model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['MSE'])

    def model_fit(self, X_train_part, y_train_part, earlyStop, epochs, validation_split):
        """
        Fit model on data
        :param X_train_part: array of inputs [# rows, # take coeffs]
        :param y_train_part: array of outputs [# rows, 1]
        :param epochs: int how many epochs for each regression
        :param validation_split: float between 0 and 1 denoting fraction of train data to use as validation
        :return: fitted model
        """
        return self.model.fit(X_train_part, y_train_part, epochs=epochs,
                              # stop when converged
                              callbacks=[earlyStop],
                              # Show logging.
                              verbose=2,
                              # Calculate validation results on validation split fraction of the training data.
                              validation_split=validation_split)

    def get_weights(self):
        """
        Save all weights of the model
        :return: tf variable with all the weights of the model
        """
        return self.model.weights

    def predictor(self, X):
        """
        Predict outputs for inputs X
        :param X: array of inputs [# rows, # take_coeffs]
        :return: array of predicted outputs [# rows, 1]
        """
        return self.model.predict(X)


class LN_mean(regularizers.Regularizer):

    def __init__(self, strength, N):
        """
        Initiate variables for regression regularizer
        :param strength: float, to get as multiplicative constant inside loss function to decrease size of coeffs vec.
        :param N: float, metric for the size of coeffs vec i.e. 2=euclidean
        """
        self.strength = strength
        self.N = N

    def __call__(self, x):
        """
        Regularize. this is being added to loss calculation
        :param x: tf variable containing weights of the model - either of weights classification or coeffs of ODE
        :return: tf variable in the form of loss
        """
        return self.strength * tf.reduce_mean(tf.abs(x) ** self.N) ** (1/self.N)


class LN(regularizers.Regularizer):

    def __init__(self, strength, N):
        """
        Initiate variables for regression regularizer
        :param strength: float, to get as multiplicative constant inside loss function to decrease size of coeffs vec.
        :param N: float, metric for the size of coeffs vec i.e. 2=euclidean
        """
        self.strength = strength
        self.N = N

    def __call__(self, x):
        """
        Regularize. this is being added to loss calculation
        :param x: tf variable containing weights of the model - either of weights classification or coeffs of ODE
        :return: tf variable in the form of loss
        """
        return self.strength * tf.reduce_sum(tf.abs(x) ** self.N) ** (1/self.N)


class LN_nonzero(regularizers.Regularizer):

    def __init__(self, strength, thresh):
        """
        Initiate variables for regression regularizer
        :param strength: float, to get as multiplicative constant inside loss function to decrease size of coeffs vec.
        :param thresh: float, threshold under which coefficients are regarded as zero for this regularizer
        """
        self.strength = strength
        self.thresh = thresh

    def __call__(self, x):
        """
        Regularize. this is being added to loss calculation
        :param x: tf variable containing weights of the model - either of weights classification or coeffs of ODE
        :return: tf variable in the form of loss
        """
        # x_nullified = copy.copy(x)
        # x_nullified[tf.abs(x_nullified) < self.thresh] = 0
        return self.strength * tf.reduce_mean(tf.math.count_nonzero(x, dtype=tf.float32))


# # NOT IN USE

# elif type == 'total_switch':
# inputs = tf.keras.Input(shape=data_shape, name="dataset")
# inputs_norm = normalizer(inputs[:, :take_coeffs])  # if only part of coefficients are desired
# switch = layers.Dense(3, use_bias=False, activation=custom_max)(inputs_norm)
#
# x1 = layers.Dense(1, use_bias=False)(inputs_norm)
# x2 = layers.Dense(1, use_bias=False)(inputs_norm)
# x3 = layers.Dense(1, use_bias=False)(inputs_norm)
# x = layers.Concatenate(axis=1)([x1, x2, x3])
#
# outputs = tf.keras.layers.Dot(axes=1)([x, switch])
#
# self.model = tf.keras.Model(inputs, outputs, name="total_switch_model")
