#%%  IMPORTS
import tensorflow as tf
import numpy as np
import pickle
import copy
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from tensorflow.python import keras
from keras import layers, regularizers
from keras.optimizers import SGD, Adam
import scipy.signal as sgn

# import my functions
import get_data_integ
import extras
import models_May2022 as models

#%% ----------------------------------------GLOBAL VARIABLES-------------------------------------------------------
global window_len, polyorder, epochs, mode, reg_type, period, part, learning_rate, decay_rate, num_switches
global comb, file_name, take_every, NReg, thresh_ODE, thresh_class, predict_until, times, use_bias, keepnorm
global beta_softmax, regularizer

# file_name = 'C:\\Users\\HP\\Desktop\\work\\Automation\\PDE-FIND\\PDE_FIND_take2\\Oct_2021_time_Vin_VD_header.csv'
# file_name = 'C:\\Users\\HP\\Desktop\\work\\Automation\\PDE-FIND\\PDE_FIND_take2\\a_numerical_period2bif.csv'
# file_name = 'C:\\Users\\HP\\Desktop\\work\\Automation\\PDE-FIND\\PDE_FIND_take2\\T_V_in_V_R_pd.csv'
# file_name = 'E:\Work\RLD\data sets\Lab 2 on resistor\July_2021_try\T_V_in_V_R_pd.csv'
# file_name = '/content/drive/MyDrive/grad_des_colab/RLD_data/T_V_in_V_R_pd.csv'
file_name = 'G:\\My Drive\\SWANN\\Pycharm Project\\a_numerical_period2bif_shorter.csv'
window_len, polyorder = 25, 1  # for Savitzky Golay filter on data, window_len must be odd
# epochs = 8000  # for every training loop in first phase. later phases are usually longer
epochs = 20000
batch_size = 1200
# mode = 'real'  # measurements from the RLD circuit or numerical calculation of chaotic Duffing oscillator
mode = 'period doubling bif'
# reg_type = 'softmax_switch_double_sparse'
reg_type = 'mysoftmax_switch_double_sparse'
part = 1  # how much of the data should be trained upon
num_switches = 2  # how many switches the model will have
keepnorm = 0
comb = 3  # how many combinations for coefficients
take_every = 1
# period = 833  # lab b old data (modulated)
# period = int(1000 / take_every)  # lab b 2021 stuff with Shani (not modulated)
period = 0
N_reg = 1/3  # root power for nullifying coefficients for L0 norm in regularizer
thresh_ODE = 1e-3  # threshold under which value of ODE coefficients will be nullified
thresh_class = 1e-3  # threshold under which value of classification coefficients will be nullified
# thresh = 0
predict_until = 20000  # how much data points to predict in "full prediction"
times = 16  # how many times to reg under same model architecture
use_bias = False  # Should the model use a bias in the regression
regularizer = 1  # 1 for regression with regularizer, 0 for w/out
beta_softmax = 3  # prefactor indisde exponent for custom activation function "mysoftmax"

#%% ---------------------------------- Models (Classes) ---------------------------------------------------------------


def predict_by_weight(dataset, wts, bias, normalizer, softmax_vec=[], thresh=0, keepnorm=1):
    """
    zero the weights with absolute value below a threshold to shorten the ODE and make prediction with it
    :param dataset: array of long dataset including the derivative in time size [whole dataset, take coeffs + 1]
    :param softmax_vec: vector of softmax predictions sizes [predict until, number of switches]
    :param wts: array of coeffs of ODE as weights. size [take coeffs, number of switches]
    :param thresh: float specifying under which value the weight should be considered zero
    :param keepnorm: 0 for nullifying what's below threshold, 1 for making final weights vec the same length as initial
                     default is normalizing
    :return: vector of predicted data in time using sparse weights.
    """

    norm_data = copy.copy(dataset[:, :n_coeffs])
    norm_data[:, :n_coeffs] = normalizer(dataset[:, :n_coeffs])
    # norm_data = dataset[:, :-1]
    # norm_data = norm_data[:predict_until, :]  # shorten vector only after normalization, dunno why

    if keepnorm:  # save old wts vec norm
        old_norm = np.linalg.norm(wts)

    if len(softmax_vec) == 0:  # for LR
        wts[abs(wts) < thresh] = 0
        if keepnorm:  # normalize wts vec to keep its initial norm
            wts = wts * old_norm / np.linalg.norm(wts)
        out = np.matmul(norm_data, wts[1:]) + wts[0]

    else:  # if LR was not used, no need to normalize, due to shitty normalizer I built
        if thresh == 0:  # no zeroing of wts
            first_step = np.matmul(norm_data, wts)
            after_softmax = np.sum(softmax_vec*first_step[:predict_until, :], axis=1)
            # after_softmax = np.sum(first_step, axis=1)
            out = after_softmax
        else:
            wts[abs(wts) < thresh] = 0
            if keepnorm:  # normalize wts vec to keep its initial norm
                wts = wts * old_norm / np.linalg.norm(wts)
            first_step = np.matmul(norm_data, wts) + bias
            after_softmax = np.sum(softmax_vec*first_step[:predict_until, :], axis=1)
            # after_softmax = np.sum(first_step, axis=1)
            out = after_softmax

    return out


def plot_loss(history, ax):
    """
    plotting the loss of a model fit as function of epoch in the training procedure
    :param history: fitted model
    :param ax: axis of fig to plot upon
    :return: plot of loss a.f. of epoch
    """
    ax.semilogy(history.history['loss'], label='loss')
    ax.semilogy(history.history['val_loss'], label='val_loss')
    ax.set_ylim([0, 1])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error [MPG]')
    ax.legend()
    ax.grid(True)


def plot_weights(ax, weight, ind):

    if len(weight) == 7:
        if ind == 0:
            ax[0, 0].plot(np.asarray(weight[3]), '.')
            ax[1, 0].plot(np.asarray(weight[3]), '.')
            ax[2, 0].plot(np.asarray(weight[3]), '.')
            ax[0, 1].plot(np.asarray(weight[3]), '.')
            ax[1, 1].plot(np.asarray(weight[3]), '.')
            ax[2, 1].plot(np.asarray(weight[3]), '.')
            ax[0, 2].plot(np.asarray(weight[3]), '.')
            ax[1, 2].plot(np.asarray(weight[3]), '.')
            ax[2, 2].plot(np.asarray(weight[3]), '.')
        elif ind == 1:
            ax[0, 0].plot(np.asarray(weight[3]), '.')
            ax[1, 0].plot(np.asarray(weight[3]), '.')
            ax[2, 0].plot(np.asarray(weight[3]), '.')
            ax[0, 1].plot(np.asarray(weight[4]), '.')
            ax[1, 1].plot(np.asarray(weight[4]), '.')
            ax[2, 1].plot(np.asarray(weight[4]), '.')
            ax[0, 2].plot(np.asarray(weight[5]), '.')
            ax[1, 2].plot(np.asarray(weight[5]), '.')
            ax[2, 2].plot(np.asarray(weight[5]), '.')
        elif ind == 2:
            ax[0, 0].plot(np.asarray(weight[3]), '.')
            ax[1, 0].plot(np.asarray(weight[4]), '.')
            ax[2, 0].plot(np.asarray(weight[5]), '.')
            ax[0, 1].plot(np.asarray(weight[3]), '.')
            ax[1, 1].plot(np.asarray(weight[4]), '.')
            ax[2, 1].plot(np.asarray(weight[5]), '.')
            ax[0, 2].plot(np.asarray(weight[3]), '.')
            ax[1, 2].plot(np.asarray(weight[4]), '.')
            ax[2, 2].plot(np.asarray(weight[5]), '.')

    elif len(weight) == 6:
        if ind == 0:
            ax[0, 0].plot(np.asarray(weight[3]), '.')
            ax[1, 0].plot(np.asarray(weight[3]), '.')
            ax[2, 0].plot(np.asarray(weight[3]), '.')
            ax[0, 1].plot(np.asarray(weight[3]), '.')
            ax[1, 1].plot(np.asarray(weight[3]), '.')
            ax[2, 1].plot(np.asarray(weight[3]), '.')
        elif ind == 1:
            ax[0, 0].plot(np.asarray(weight[3]), '.')
            ax[1, 0].plot(np.asarray(weight[3]), '.')
            ax[2, 0].plot(np.asarray(weight[3]), '.')
            ax[0, 1].plot(np.asarray(weight[4]), '.')
            ax[1, 1].plot(np.asarray(weight[4]), '.')
            ax[2, 1].plot(np.asarray(weight[4]), '.')
        elif ind == 2:
            ax[0, 0].plot(np.asarray(weight[3]), '.')
            ax[1, 0].plot(np.asarray(weight[4]), '.')
            ax[2, 0].plot(np.asarray(weight[3]), '.')
            ax[0, 1].plot(np.asarray(weight[4]), '.')
            ax[1, 1].plot(np.asarray(weight[3]), '.')
            ax[2, 1].plot(np.asarray(weight[4]), '.')

    else:
        ax.plot(np.asarray(weight[3]), '.')


def get_softmax_vec(dataset_lng, wts_classif, bias, normalizer):
    wts_len = wts_classif.shape[0]
    dataset_norm = copy.copy(dataset_lng[:, :n_coeffs])
    # dataset_norm[:, 1:n_coeffs] = normalizer(dataset_lng[:, 1:n_coeffs])  # old
    dataset_norm[:, 0:n_coeffs] = normalizer(dataset_lng[:, 0:n_coeffs])  # new
    # dataset_norm = np.append(dataset_norm, np.array([dataset_lng[:,-1]]).T, axis=1)
    # dataset_norm = dataset_lng

    wts_for_softmax = copy.copy(wts_classif)
    wts_for_softmax[abs(wts_for_softmax) < thresh_class] = 0  # nullify what's below the threshold
    for_softmax = np.empty([dataset_norm.shape[0], num_switches])
    softmax_vec = np.empty([dataset_norm.shape[0], num_switches])

    for i in range(dataset_norm.shape[0]):
        for j in range(num_switches):
            for_softmax[i, j] = np.dot(dataset_norm[i, :wts_len], wts_for_softmax[:, j]) + bias[j]
        softmax_vec[i, :] = np.exp(for_softmax[i, :]) / np.sum(np.exp(for_softmax[i, :]))

    return softmax_vec


def plot_classification(dataset_norm, dVdt, softmax_vec):
    """
    Plot the input data with coloring denoting where each model was the strongest.
    :param dataset_norm: array containing all data including combinations [#rows, #take coeffs]
    :param :
    :return: 1) figure of data and where each model was more dominant in colors
             2) array of the softmax results for all data points and for each model [#rows, #models]
    """
    color_str = ['blue', 'pink', 'red', 'green', 'orange', 'purple']
    fig4, ax4 = plt.subplots()
    plt.plot(dVdt, 'k')
    for i in range(num_switches):
        ax4.fill_between(range(dataset_norm.shape[0]), 0, 1, where=softmax_vec[:, i] == np.max(softmax_vec, axis=1),
                         color=color_str[i], alpha=0.3, transform=ax4.get_xaxis_transform())

    return fig4, ax4


def get_switch_vec_etc(softmax_vec):

    switches_vec = np.zeros([len(softmax_vec), num_switches])
    for k in range(num_switches):
        switches_vec[:, k] = (softmax_vec[:, k] == np.max(softmax_vec, axis=1)).astype(int)
    switch_indices = np.where(np.sum(np.abs(np.diff(switches_vec, axis=0)), axis=1))
    delta_switches = switch_indices[0][1:] - switch_indices[0][:-1]

    return switches_vec, switch_indices, delta_switches


def plot_switches(delta_switches):
    fig3, ax3 = plt.subplots(1, 2)
    ax3[0].plot(delta_switches)
    ax3[0].set_xlabel('switch num')
    ax3[0].set_ylabel('dT between switches')
    ax3[1].hist(delta_switches)
    ax3[1].set_xlabel('dT between switches')
    ax3[1].set_ylabel('count')
    return fig3, ax3


def reg(type, dataset, normalizer, optimizer, regularizer, earlyStop):

    # normalize all datasets using this normalizer
    # X_train = normalizer(X_train)
    # X_test = normalizer(X_train)
    # dataset_norm = normalizer(dataset_lng[:, :-1])
    # dataset_norm = np.append(dataset_norm, np.array([dataset_lng[:, -1]]))
    # dataset_norm = copy.copy(dataset)
    dataset_norm = normalizer(dataset[:, :-1])

    tf.keras.backend.clear_session()
    # model = models.MyModel(type, num_switches, normalizer, [int(dataset_norm.shape[1] - 1), n_coeffs],
    #                        regularizer, use_bias=use_bias)
    model = models.MyModel(reg_type, num_switches, normalizer, [int(dataset_norm.shape[1]), n_coeffs],
                           regularizer_ODE=0, regularizer_classif=0, use_bias=use_bias)

    # split into training and test
    X_train, X_test, y_train, y_test = train_test_split(dataset[:, :-1], dataset[:, -1],
                                                        test_size=0.20, shuffle=True)
    # X_train, X_test, y_train, y_test = train_test_split(dataset_norm, dataset[:, -1],
    #                                                     test_size=0.20, shuffle=True)

    # adapt normalizer
    # normalizer.adapt(X_train)

    # compile model with new shuffled data
    model.model_compile(optimizer=optimizer)

    # fit
    history = model.model_fit(X_train[:int(part * len(X_train)), :], y_train[:int(part * len(X_train))],
                              earlyStop, epochs=epochs, validation_split=0.2)

    weight = model.get_weights()
    wts_means = weight[0].numpy()
    wts_var = weight[1].numpy()
    bias = wts_means[0] - wts_means[1] - np.sqrt(0.01) * 1/3
    C_VIN_VR = np.sqrt(wts_var[0])
    C_VR3 = -np.sqrt(wts_var[1])
    dynamics = C_VIN_VR * dataset_norm[:, 0] + C_VR3 * dataset_norm[:, 1] + bias
    plt.figure()
    plt.plot(dynamics)
    plt.plot(dataset[:, -1])

    train_predictions = model.predictor(X_train).flatten()
    test_predictions = model.predictor(X_test).flatten()
    all_pred_by_time = model.predictor(dataset[:predict_until, :-1]).flatten()

    # note R squared
    r2 = r2_score(y_test, test_predictions)
    print("R2 score on training set is: {:0.3f}".format(r2_score(y_train, train_predictions)))
    print("R2 score on test set is: {:0.3f}".format(r2))

    return weight, train_predictions, test_predictions, all_pred_by_time, history, r2

#%% MAIN

# ----------------------------------------- GET DATA IN FORM OF ARRAY etc. -----------------------------------------

# get data in form of dataframe from csv file
df = get_data_integ.build_df_noQ(mode, win_len=window_len, porder=polyorder, file=file_name, demean=1,
                                     take_every=take_every)

# df = df.iloc[:int(7500 / take_every)]  # old
# df = df.iloc[int(7500 / take_every)+500:2*int(7500 / take_every)]  # new
df = df.iloc[:2*int(8000 / take_every)]  # new longer

# calculate dyamics, AKA derivative in time
V_Rt = extras.good_derivative(df['V_R'].values, df['dt'].values, order=8)
# V_Rt = np.convolve(V_Rt, np.ones(4), 'valid') / 4
V_Rt = savgol_filter(V_Rt, window_length=25, polyorder=1)

# name all desired fields in the dataset including readings of the data few periods backwards
# arr_for_comb = [np.ones(len(df['V_R'])), df['V_R'], df['V_in'], np.roll(df['V_R'], period),
#                 np.roll(df['V_R'], period * 2), np.roll(df['V_R'], period * 3),
#                 np.roll(df['V_R'], period * 4), np.roll(df['V_R'], period * 5)]
arr_for_comb = [np.ones(len(df['V_R'])), df['T'], df['V_R']]
# arr_for_comb_verbal = ['1', 'V_in', 'V_R', 'V_R_1back', 'V_R_2back', 'V_R_3back', 'V_R_4back', 'V_R_5back']
arr_for_comb_verbal = ['1', 'V_in', 'V_R']
# arr_for_comb = [np.ones(len(df['V_R'])), df['V_R'], df['Q'], df['V_in'], env_V_in, np.roll(df['V_R'], period),
#                 np.roll(df['Q'], period), np.roll(df['V_R'], period * 2), np.roll(df['Q'], period * 2),
#                 np.roll(df['V_R'], period * 3), np.roll(df['Q'], period * 3), np.roll(df['V_R'], period * 4),
#                 np.roll(df['Q'], period * 4), np.roll(df['V_R'], period * 5), np.roll(df['Q'], period * 5)]

# add desired fields to dataset with relevant multiplication combinations within them
dataset_lst = list(combinations_with_replacement(arr_for_comb, comb))
coeffs_lst = list(combinations_with_replacement(arr_for_comb_verbal, comb))
dataset_lng = np.transpose(np.array([np.prod(i, axis=0) for i in dataset_lst]))
# if dataset_lng[-1,6] > 100:
#     dataset_lng[:,6] = 0

# add V_Rt at the end as result
dataset_lng = np.concatenate((dataset_lng, np.array([V_Rt]).T), axis=1)

# give up on data on edges, corresponding to amount of periods taken back, and also on 'ones' columns due to bias
# also give up on the first column of 'ones' in favor of bias
# dataset_lng = dataset_lng[period * 5: - period * 5, 1:]
# dataset_lng = dataset_lng[150: -150, 1:]
dataset_lng = dataset_lng[150: -150, 1:]
erase_indices = np.arange(int(dataset_lng.shape[0]/2) - 50, int(dataset_lng.shape[0]/2) + 50)
dataset_lng = np.delete(dataset_lng, erase_indices, axis=0)

denoter = np.ones([len(dataset_lng[:, 0])])
denoter[:int(len(dataset_lng[:, 0])/2)] = -1

# use only relevant ones
dataset_lng = np.transpose(np.array([dataset_lng[:, 3], dataset_lng[:, 8],
                                     # dataset_lng[:, 1],
                                     denoter,
                                     dataset_lng[:, -1]]))

# dataset_lng = np.transpose(np.array([dataset_lng[:, 3], dataset_lng[:, 8],
#                                      dataset_lng[:, 0], dataset_lng[:, 1], dataset_lng[:, 2], dataset_lng[:, 4],
#                                      dataset_lng[:, 5],
#                                      dataset_lng[:, -1]]))

plt.figure()
for i in range(dataset_lng.shape[1]):
    plt.plot(dataset_lng[:,i])
plt.show()

# normalize using keras normalization
# normalizer = tf.keras.layers.Normalization(axis=-1)
# normalizer_shrt = tf.keras.layers.Normalization(axis=-1)
# normalizer.adapt(dataset_lng[:int(part * dataset_lng.shape[0]), 1:-1])  # a general adapt to see model summary
# normalizer_shrt.adapt(dataset_lng[:int(part * dataset_lng.shape[0]), 1:n_coeffs])

## ------------------------------- REGRESSION  --------------------------------
# # wts_ODE, wts_classif, normalizer_shrt = multip_LRs_different_coeffs_n_switches('softmax_switch_double_sparse',
# #                                                                                dataset_lng, part, regularizer=1)
# # wts_ODE, wts_classif, normalizer_shrt = multip_LRs_dessications('LR', dataset_lng, part, regularizer=1)
# wts_ODE, wts_classif, normalizer_shrt = multip_LRs_dessications('softmax_switch_double_sparse', dataset_lng, part, regularizer=1)
# # LR_plus_vis('softmax_switch_double_sparse', np.c_[dataset_lng[:, :take_coeffs], dataset_lng[:, -1]], sgd,
# #             part, regularizer=LNReg)
# # LR_plus_vis('LR', dataset_lng, part, regularizer=0)
# # LR_plus_vis('linear_model', np.c_[dataset_lng[:, :take_coeffs], dataset_lng[:, -1]], sgd, normalizer_shrt,
# #             num, part)

#%% --------------------------------------------------- INITS --------------------------------------------------------
# initiate lists to save the different models and their weights
# models = list()
wts_classification = list()
wts_ODE = list()

global n_coeffs
n_coeffs = len(dataset_lng[0]) - 1  # coefficients (not including bias) are the colulmns of dataset_lng except last

# for loss
fig1, ax1 = plt.subplots(1, 1)

k = 1  # iterative number for denoting the files saved
keepvals = np.linspace(0, n_coeffs - 1, n_coeffs,
                       dtype=int)  # is this how Im gonna save who enters and who kept outside?

global strength
# strength = 0.05 / (num_switches * take_coeffs) ** (1/2)  # multip. const. at loss func. for nullifying
# the coeffs vec
# strength = 10 / n_coeffs ** (1/2)  # multip. const. at loss func. for nullifying
strength = 0.005 / (n_coeffs) ** (1 / 2)

# prelims 1 - define optimizer and callbacks for early stopping of fit procedure

# prelims 2 - Optimizer
# optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, clipnorm = 1.0, nesterov=False)
# optimizer = Adam(beta_1=decay_rate)
optimizer = Adam()

# prelims 3 - Callbacks for early stopping when weights converge
earlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=840)

# prelims 4 - Regularizer
if regularizer == 1:
    regularizer = models.LN(strength, N_reg)
else:
    pass

# prelims 5 - Take only relevant terms on data

# dataset_lng = np.c_[dataset_lng[:, :n_coeffs], dataset_lng[:, -1]]
# dataset = np.c_[dataset_lng[:, :n_coeffs], dataset_lng[:, -1]]
dataset = np.c_[dataset_lng[:, keepvals], dataset_lng[:, -1]]

# normalize using keras normalization
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer_shrt = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(dataset[:int(part * dataset.shape[0]), 0:-1])
# normalizer.adapt(dataset_lng[:int(part * dataset_lng.shape[0]), 0:-1])  # general adapt to see model summary
# normalizer.adapt(dataset_lng[:int(dataset_lng.shape[0]), 0:-1])  # general adapt to see model summary
if type == 'LR':
    normalizer_shrt.adapt(dataset[:int(part * dataset.shape[0]), :n_coeffs])
    # normalizer_shrt.adapt(dataset_lng[:int(part * dataset_lng.shape[0]), :n_coeffs])
    # normalizer_shrt.adapt(dataset_lng[:int(dataset_lng.shape[0]), :take_coeffs])
else:
    normalizer_shrt.adapt(dataset[:int(part * dataset.shape[0]), :n_coeffs])
    # normalizer_shrt.adapt(dataset_lng[:int(part * dataset_lng.shape[0]), :n_coeffs])
    # normalizer_shrt.adapt(dataset_lng[:int(dataset_lng.shape[0]), :take_coeffs])

#%% ------------------------------------------------- REG ----------------------------------------------------
weight, train_predictions, test_predictions, all_pred_by_time, history, r2 = reg(type, dataset,
                                                                                  normalizer_shrt,
                                                                                  optimizer, regularizer,
                                                                                  earlyStop)

#%% ------------------------------------------------- REG 2ND STYLE------------------------------------------
K=5
X = dataset[:, :-1]
y = dataset[:, -1]

for train, test in KFold(n_splits=K, shuffle=True).split(dataset[:, :-1], dataset[:, -1]):

    dataset_norm = normalizer(dataset[:, :-1])

    tf.keras.backend.clear_session()
    model = models.MyModel(reg_type, num_switches, normalizer, [int(dataset_norm.shape[1]), n_coeffs],
                           regularizer_ODE=0, regularizer_classif=0, use_bias=use_bias)

    # compile model with new shuffled data
    model.model_compile(optimizer=optimizer)

    # fit
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=840)
    history = model.model.fit(X[train], y[train], callbacks=[earlyStop], validation_data=(X[test], y[test]),
                        epochs=epochs, batch_size=batch_size).history

    weight = model.get_weights()
    wts_means = weight[0].numpy()
    wts_var = weight[1].numpy()

    train_predictions = model.predictor(X[train]).flatten()
    test_predictions = model.predictor(X[test]).flatten()
    all_pred_by_time = model.predictor(dataset[:predict_until, :-1]).flatten()

    # note R squared
    r2 = r2_score(y[test], test_predictions)
    print("R2 score on training set is: {:0.3f}".format(r2_score(y[train], train_predictions)))
    print("R2 score on test set is: {:0.3f}".format(r2))


fig7, ax7 = plt.subplots(1,1)
ax7.plot(y[train])
ax7.plot(y[test])
plt.show()


#%% ------------------------------------------- SAVE AND PLOTS -----------------------------------------------
# add wts to lsts
if type != 'LR':
    if use_bias:
        wts_ODE.append(np.asarray(weight[5]))
        wts_classification.append(np.asarray(weight[3]))
        bias_ODE = weight[6].numpy()
        bias_classification = weight[4].numpy()
    else:
        wts_ODE.append(np.asarray(weight[4]))
        wts_classification.append((np.asarray(weight[3])))
        bias_ODE = np.zeros(2, )
        bias_classification = np.zeros(2, )
else:
    bias = weight[4].numpy()
    wts_ODE.append(np.insert(np.asarray(weight[3]), 0, bias))

# plot loss
plot_loss(history, ax1)

# save softmaxes and plot
if type != 'LR':  # if softmaxes are used
    # classify different regimes in switch model
    softmax_vec = get_softmax_vec(dataset_lng[:predict_until, :], wts_classification[-1],
                                  bias_classification, normalizer_shrt)
    fig4, ax4 = plot_classification(dataset[:predict_until, :], dataset_lng[:predict_until, -1],
                                    softmax_vec)

    # get where there are switches in data and statistics on them
    switches_vec, switch_indices, delta_switches = get_switch_vec_etc(softmax_vec)

    # plot delta between switches in time
    fig3, ax3 = plot_switches(delta_switches)

    # wts_classification[-1][abs(wts_classification[-1]) < thresh_class] = 0  # nullify classif before pred by wts
    softmax_null = get_softmax_vec(dataset_lng[:predict_until, :], wts_classification[-1], bias_classification,
                                   normalizer_shrt)
    predict_by_wts = predict_by_weight(dataset[:predict_until, :], wts_ODE[-1], bias_ODE, normalizer_shrt,
                                       softmax_vec=softmax_null, thresh=thresh_ODE)

else:  # softmax is not used
    fig4, ax4 = plt.subplots()
    fig3, ax3 = plt.subplots(1, 2)
    softmax_vec = []
    softmax_null = []
    predict_by_wts = predict_by_weight(dataset[:predict_until, :], wts_ODE[-1], bias_ODE, normalizer_shrt,
                                       thresh=thresh_ODE, keepnorm=keepnorm)

ax4.plot(dataset[:predict_until, -1], 'b')
ax4.plot(predict_by_wts, 'g')
ax4.plot(all_pred_by_time, 'r')
ax4.set_xlabel('time')
ax4.set_ylabel('dV/dt')
plt.legend(['measured', 'predicted sparse', 'predicted'])

bias_calc_up = np.mean(dataset_lng[:, 0], axis=0) - np.mean(dataset_lng[:, 1], axis=0) - np.sqrt(0.01) * 1 / 3
bias_calc_down = - np.mean(dataset_lng[:, 0], axis=0) - np.mean(dataset_lng[:, 1], axis=0) + np.sqrt(0.01) * 1 / 3
# wts_transf_back = wts_ODE[-1] / np.insert(np.std(dataset[:, :-1], axis=0), 0, bias_calc)  # old
wts_transf_back = np.array([wts_ODE[-1].T / np.std(dataset[:, :-1], axis=0)])  # new
if all(bias_ODE == 0):  # if there is no use of bias
    bias_transf_back_up = wts_ODE[-1][-1] / bias_calc_up  # two values because we can't know which model the net chose
    bias_transf_back_down = wts_ODE[-1][-1] / bias_calc_down  # same as above
else:
    bias_transf_back_up = bias_ODE / bias_calc_up
    bias_transf_back_down = bias_ODE / bias_calc_down

dataset_norm = normalizer(dataset[:, :-1])
# dynamics2 = wts_ODE[0][1] * dataset_norm[:, 0] + wts_ODE[0][2] * dataset_norm[:, 1] + wts_ODE[0][0]
dynamics2 = np.asarray([wts_ODE[0][0]]) * np.asarray([dataset_norm[:, 0]]).T \
            + np.asarray([wts_ODE[0][1]]) * np.asarray([dataset_norm[:, 1]]).T \
            + np.asarray([wts_ODE[0][2]]) * np.asarray([dataset_norm[:, 2]]).T \
            + bias_ODE

plt.figure()
plt.plot(dynamics2[:, 0] - dataset[:, -1])
plt.plot(dynamics2[:, 1] - dataset[:, -1])

plt.show()
fig3.show()
fig4.show()

#%% Extra plots to see if one model got good fit

fig6, ax6 = plt.subplots(1, 1)
# ax6.plot(-dataset_lng[:,0]*0.4535 - dataset_lng[:,1]*0.457 + dataset_lng[:,2]*0.107 + 1/10, 'r')
after_soft1 = np.matmul(dataset_lng[:, :-1], wts_transf_back[:, 0].T)
real_bias1 = np.sum(wts_ODE[-1][:, 0] * np.mean(dataset_lng[:, :-1], axis=0) / np.std(dataset_lng[:, :-1], axis=0))
after_soft2 = np.matmul(dataset_lng[:, :-1], wts_transf_back[:, 1].T)
real_bias2 = np.sum(wts_ODE[-1][:, 1] * np.mean(dataset_lng[:, :-1], axis=0) / np.std(dataset_lng[:, :-1], axis=0))
dataset_from_unnormalized = (after_soft1 - real_bias1) * np.array([softmax_vec[:, 0]]).T\
                            + 0
                            # + (after_soft2 - real_bias2) * np.array([softmax_vec[:, 1]]).T


ax6.plot(np.matmul(dataset_lng[:, :-1], wts_transf_back[:, 0].T), 'r')
ax6.plot(dataset_from_unnormalized, 'b')
ax6.plot(dataset_lng[:, -1], 'g')
ax6.set_xlim([(np.floor(len(dataset_lng))/2).astype(int), len(dataset_lng)])
ax6.set_ylim([-1, 2.5])
fig6.show()

#%% Calculate goodness of regularization

post_regularizer1 = strength * regularizer(wts_ODE[-1][:, 0])
post_regularizer2 = strength * regularizer(wts_ODE[-1][:, 1])
general_loss = np.mean(np.abs(all_pred_by_time - dataset_lng[:,-1]))

#%% Save
# from prediction after nullification
r2_after_nullify = r2_score(dataset[:predict_until, -1], predict_by_wts)
print("R2 after nullifying on beginning of data is: {:0.3f}".format(r2_after_nullify))

pickle.dump(fig1, open('loss_' + str(n_coeffs) + 'coeffs_' + str(num_switches) + 'switches.pickle', 'wb'))
pickle.dump(fig3, open('dT_by_time_and_hist_' + str(n_coeffs) + 'coeffs_' + str(num_switches) +
                       'switches' + str(k) + '.pickle', 'wb'))
pickle.dump(fig4, open('dynamics_measured_vs_ANN_' + str(n_coeffs) + 'coeffs_' + str(num_switches) +
                       'switches' + str(k) + '.pickle', 'wb'))
pickle.dump(softmax_vec, open('softmax_vec', 'wb'))
pickle.dump(wts_ODE, open('wts_ODE', 'wb'))
pickle.dump(wts_classification, open('wts_classification', 'wb'))
pickle.dump(bias_ODE, open('bias_ODE', 'wb'))
pickle.dump(bias_classification, open('bias_classification', 'wb'))
pickle.dump(predict_by_wts, open('predict_by_weights', 'wb'))
pickle.dump(all_pred_by_time, open('predict_by_ANN', 'wb'))

k += 1
keepvals = np.where(wts_ODE[-1][1:] != 0)[0]
n_coeffs = len(keepvals)
# wts_classification[-1][abs(wts_classification[-1]) < thresh] = 0

# pickle.dump(fig2, open('weights_in_order.pickle', 'wb'))

now = datetime.now()
dt_string = now.strftime("%d.%m.%Y_%H%M%S")
np.savetxt("r2_" + str(dt_string) + ".csv", r2, delimiter=",")
np.savetxt("r2_nullified_" + str(dt_string) + ".csv", r2_after_nullify, delimiter=",")
np.savetxt("num_of_switches_" + str(dt_string) + ".csv", num_switches, delimiter=",")
np.savetxt("num_of_coeffs_in_every_switch_" + str(dt_string) + ".csv", n_coeffs, delimiter=",")


# -------------------------------------- PREDICTION USING MEAN OF WEIGHTS ------------------------------------------

wts_ODE_arr = np.concatenate([wts_ODE[i] for i in range(len(wts_ODE))], axis=1)
wts_classif_arr = np.concatenate([wts_ODE[i] for i in range(len(wts_ODE))], axis=1)
wts_all_arr = np.concatenate([wts_ODE_arr, wts_classif_arr], axis=0)

kmeans = KMeans(n_clusters=num_switches, init='k-means++', max_iter=300, n_init=10)
y_pred = kmeans.fit_predict(wts_all_arr.T)

c_centres = kmeans.cluster_centers_.T
wts_ODE_mean = c_centres[:20, :]
wts_classif_mean = c_centres[20:, :]

softmax_after_null = get_softmax_vec(dataset_lng[:predict_until], wts_classif_mean, normalizer_shrt)
pred = predict_by_weight(dataset_lng[:predict_until, :-1], wts_ODE_mean, normalizer_shrt,
                         softmax_vec=softmax_after_null, thresh=thresh_ODE)

# --------------------------------- MULTIPLE REGRESSIONS TO SEE SPREAD OF COEFFS -----------------------------------

# multip_reg_for_weight_spread('softmax_switch_double_sparse', dataset_lng, part, times, regularizer=1)

# how to open the file that was saved with 'dump' saved before
# pickle.load(open('dynamics_by_type' + str(take_coeffs) + '.pickle', 'rb'))

#%% NOT IN USE

# def multip_LRs_dessications(type, dataset_lng, part, regularizer=0):
#     """
#     produces the regression and visualizes it
#     :param type: string naming the desired model to check
#     :param dataset_lng: the full dataset as numpy array - columns are fields and rows are datapoints in time
#     :param part: float between 0 and 1 denoting the fraction of data to use in train and test
#     :param regularizer: specify 1 if regularizer needed for ridge/lasso regression if desired, 0 if not
#     :return: plots of different kinds, specified inside
#     """
#
#     # --------------------------------------------------- INITS --------------------------------------------------------
#     # initiate lists to save the different models and their weights
#     # models = list()
#     wts_classification = list()
#     wts_ODE = list()
#
#     global n_coeffs
#     n_coeffs = len(dataset_lng[0]) - 1  # coefficients (not including bias) are the colulmns of dataset_lng except last
#
#     # for loss
#     fig1, ax1 = plt.subplots(1, 1)
#
#     # --------------------------------------------------- LOOP ---------------------------------------------------------
#     # iterate over switches and coeffs
#
#     k = 1  # iterative number for denoting the files saved
#     keepvals = np.linspace(0, n_coeffs - 1, n_coeffs, dtype=int)  # is this how Im gonna save who enters and who kept outside?
#
#     for i in range(dessications):
#         global strength
#         # strength = 0.05 / (num_switches * take_coeffs) ** (1/2)  # multip. const. at loss func. for nullifying
#         # the coeffs vec
#         # strength = 10 / n_coeffs ** (1/2)  # multip. const. at loss func. for nullifying
#         strength = 0.005 / (n_coeffs) ** (1/2)
#
#         # prelims 1 - define optimizer and callbacks for early stopping of fit procedure
#
#         # prelims 2 - Optimizer
#         # optimizer = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, clipnorm = 1.0, nesterov=False)
#         # optimizer = Adam(beta_1=decay_rate)
#         optimizer = Adam()
#
#         # prelims 3 - Callbacks for early stopping when weights converge
#         earlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=840)
#
#         # prelims 4 - Regularizer
#         if regularizer == 1:
#             regularizer = models.LN(strength, N_reg)
#         else:
#             pass
#
#         # prelims 5 - Take only relevant terms on data
#
#         # dataset_lng = np.c_[dataset_lng[:, :n_coeffs], dataset_lng[:, -1]]
#         # dataset = np.c_[dataset_lng[:, :n_coeffs], dataset_lng[:, -1]]
#         dataset = np.c_[dataset_lng[:, keepvals], dataset_lng[:, -1]]
#
#         # normalize using keras normalization
#         normalizer = tf.keras.layers.Normalization(axis=-1)
#         normalizer_shrt = tf.keras.layers.Normalization(axis=-1)
#         normalizer.adapt(dataset[:int(part * dataset.shape[0]), 0:-1])
#         # normalizer.adapt(dataset_lng[:int(part * dataset_lng.shape[0]), 0:-1])  # general adapt to see model summary
#         # normalizer.adapt(dataset_lng[:int(dataset_lng.shape[0]), 0:-1])  # general adapt to see model summary
#         if type == 'LR':
#             normalizer_shrt.adapt(dataset[:int(part * dataset.shape[0]), :n_coeffs])
#             # normalizer_shrt.adapt(dataset_lng[:int(part * dataset_lng.shape[0]), :n_coeffs])
#             # normalizer_shrt.adapt(dataset_lng[:int(dataset_lng.shape[0]), :take_coeffs])
#         else:
#             normalizer_shrt.adapt(dataset[:int(part * dataset.shape[0]), :n_coeffs])
#             # normalizer_shrt.adapt(dataset_lng[:int(part * dataset_lng.shape[0]), :n_coeffs])
#             # normalizer_shrt.adapt(dataset_lng[:int(dataset_lng.shape[0]), :take_coeffs])
#
#         # ------------------------------------------------- REG ----------------------------------------------------
#         weight, train_predictions, test_predictions, all_pred_by_time, history, r2i = reg(type, dataset,
#                                                                                           normalizer_shrt,
#                                                                                           optimizer, regularizer,
#                                                                                           earlyStop)
#
#         # ------------------------------------------- SAVE AND PLOTS -----------------------------------------------
#         # add wts to lsts
#         if type != 'LR':
#             if use_bias:
#                 wts_ODE.append(np.asarray(weight[5]))
#                 wts_classification.append(np.asarray(weight[3]))
#                 bias_ODE = weight[6].numpy()
#                 bias_classification = weight[4].numpy()
#             else:
#                 wts_ODE.append(np.asarray(weight[4]))
#                 wts_classification.append((np.asarray(weight[3])))
#                 bias_ODE = np.zeros(2,)
#                 bias_classification = np.zeros(2,)
#         else:
#             bias = weight[4].numpy()
#             wts_ODE.append(np.insert(np.asarray(weight[3]), 0, bias))
#
#         # plot loss of every iteration
#         plot_loss(history, ax1[i])
#
#         # save softmaxes and plot
#         if type != 'LR':  # if softmaxes are used
#             # classify different regimes in switch model
#             softmax_vec = get_softmax_vec(dataset_lng[:predict_until, :], wts_classification[-1],
#                                           bias_classification, normalizer_shrt)
#             fig4, ax4 = plot_classification(dataset[:predict_until, :], dataset_lng[:predict_until, -1],
#                                             softmax_vec)
#
#             # get where there are switches in data and statistics on them
#             switches_vec, switch_indices, delta_switches = get_switch_vec_etc(softmax_vec)
#
#             # plot delta between switches in time
#             fig3, ax3 = plot_switches(delta_switches)
#
#             # wts_classification[-1][abs(wts_classification[-1]) < thresh_class] = 0  # nullify classif before pred by wts
#             softmax_null = get_softmax_vec(dataset_lng[:predict_until, :], wts_classification[-1], bias_classification,
#                                            normalizer_shrt)
#             predict_by_wts = predict_by_weight(dataset[:predict_until, :], wts_ODE[-1], bias_ODE, normalizer_shrt,
#                                                softmax_vec=softmax_null, thresh=thresh_ODE)
#
#         else:  # softmax is not used
#             fig4, ax4 = plt.subplots()
#             fig3, ax3 = plt.subplots(1, 2)
#             softmax_vec = []
#             softmax_null = []
#             predict_by_wts = predict_by_weight(dataset[:predict_until, :], wts_ODE[-1], bias_ODE, normalizer_shrt,
#                                                thresh=thresh_ODE, keepnorm=keepnorm)
#
#         ax4.plot(dataset[:predict_until, -1], 'b')
#         ax4.plot(predict_by_wts, 'g')
#         ax4.plot(all_pred_by_time, 'r')
#         ax4.set_xlabel('time')
#         ax4.set_ylabel('dV/dt')
#         plt.legend(['measured', 'predicted sparse', 'predicted'])
#
#         bias_calc = np.mean(dataset_lng[:, 0], axis=0) - np.mean(dataset_lng[:, 1], axis=0) - np.sqrt(0.01) * 1/3
#         # wts_transf_back = wts_ODE[-1] / np.insert(np.std(dataset[:, :-1], axis=0), 0, bias_calc)  # old
#         wts_transf_back = np.array([wts_ODE[-1].T / np.std(dataset[:, :-1], axis=0)])  # new
#         bias_transf_back = bias_ODE / bias_calc
#
#         # from prediction without total nullification
#         r2[i] = r2i
#
#         dataset_norm = normalizer(dataset[:, :-1])
#         # dynamics2 = wts_ODE[0][1] * dataset_norm[:, 0] + wts_ODE[0][2] * dataset_norm[:, 1] + wts_ODE[0][0]
#         dynamics2 = np.asarray([wts_ODE[0][0]]) * np.asarray([dataset_norm[:, 0]]).T\
#                     + np.asarray([wts_ODE[0][1]]) * np.asarray([dataset_norm[:, 1]]).T \
#                     + np.asarray([wts_ODE[0][2]]) * np.asarray([dataset_norm[:, 2]]).T \
#                     + bias_ODE
#
#         plt.figure()
#         plt.plot(dynamics2[:,0] - dataset[:,-1])
#         plt.plot(dynamics2[:,1] - dataset[:,-1])
#
#         # from prediction after nullification
#         r2_after_nullify = r2_score(dataset[:predict_until, -1], predict_by_wts)
#         print("R2 after nullifying on beginning of data is: {:0.3f}".format(r2_after_nullify[i]))
#
#         pickle.dump(fig1, open('loss_' + str(n_coeffs) +'coeffs_' + str(num_switches) + 'switches.pickle', 'wb'))
#         pickle.dump(fig3, open('dT_by_time_and_hist_' + str(n_coeffs) +'coeffs_' + str(num_switches) +
#                                'switches' + str(k) + '.pickle', 'wb'))
#         pickle.dump(fig4, open('dynamics_measured_vs_ANN_' + str(n_coeffs) +'coeffs_' + str(num_switches) +
#                                'switches' + str(k) + '.pickle', 'wb'))
#         pickle.dump(softmax_vec, open('softmax_vec', 'wb'))
#         pickle.dump(wts_ODE, open('wts_ODE', 'wb'))
#         pickle.dump(wts_classification, open('wts_classification', 'wb'))
#         pickle.dump(bias_ODE, open('bias_ODE', 'wb'))
#         pickle.dump(bias_classification, open('bias_classification', 'wb'))
#         pickle.dump(predict_by_wts, open('predict_by_weights', 'wb'))
#         pickle.dump(all_pred_by_time, open('predict_by_ANN', 'wb'))
#
#         k += 1
#         keepvals = np.where(wts_ODE[-1][1:] != 0)[0]
#         n_coeffs = len(keepvals)
#         # wts_classification[-1][abs(wts_classification[-1]) < thresh] = 0
#
#     # pickle.dump(fig2, open('weights_in_order.pickle', 'wb'))
#
#     now = datetime.now()
#     dt_string = now.strftime("%d.%m.%Y_%H%M%S")
#     np.savetxt("r2_" + str(dt_string) + ".csv", r2, delimiter=",")
#     np.savetxt("r2_nullified_" + str(dt_string) + ".csv", r2_after_nullify, delimiter=",")
#     np.savetxt("num_of_switches_" + str(dt_string) + ".csv", num_switches, delimiter=",")
#     np.savetxt("num_of_coeffs_in_every_switch_" + str(dt_string) + ".csv", n_coeffs, delimiter=",")
#
#     return wts_ODE, wts_classification, normalizer_shrt
