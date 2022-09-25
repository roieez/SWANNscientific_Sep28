# IMPORTS

# general
import copy
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

# my functions
import duffing_dynamics
import calculators
import extras


def get_data_pd(dynamics, win_len=0, porder=0, file=0, take_every=1):

    if dynamics == 'real2':
        df = pd.read_csv(file)

        df = df.iloc[::take_every, :]

        # Savitzky Golay filter from user parameters - window length and poly-order
        df['V_R'] = savgol_filter(df['V_R'].values, window_length=win_len, polyorder=porder)
        df['V_in'] = savgol_filter(df['V_in'].values, window_length=win_len, polyorder=porder)

        df['T'], df['V_R'], df['V_in'] = normalize(df['T'].values, df['V_R'].values, df['V_in'].values)

        # dt = df['T'].values[1:] - df['T'].values[0:-1]
        # dt = np.append(dt, dt[-1])
        # dt[dt == 0] = np.mean(dt)
        # df['dt'] = dt
        # if file =
        df['dt'] = np.ones(len(df['V_R'].values)) * np.mean(df['T'].values[1:] - df['T'].values[0:-1])
        if np.isnan(df['dt'].iloc[1]):
            dt = 4*10**(-8)
            df['dt'] = np.ones(len(df['V_R'].values)) * dt
            df['T'] = np.arange(0, dt*len(df['V_R'].values), dt)

    elif dynamics == 'period doubling bif':
        df = pd.read_csv(file)

        # down sample / coarse grain
        df = df.iloc[::take_every, :]

        # df['dt'] = np.ones(len(df['V_R'].values)) * (df['T'].values[1] - df['T'].values[0])
        dt_partial = df['T'].values[1:] - df['T'].values[0:-1]
        df['dt'] = np.append(dt_partial, dt_partial[0])
        # df['T'], df['V_R'], df['V_in'] = normalize(df['T'].values, df['V_R'].values, df['V_in'].values)
        if np.isnan(df['dt'].iloc[1]):
            dt = 0.01
            df['dt'] = np.ones(len(df['V_R'].values)) * dt
            df['T'] = np.arange(0, dt*len(df['V_R'].values), dt)

    return df




def build_df(mode, detrend=1, win_len=0, porder=0, file=0, demean=0):
    """
    builds Pandas DataFrame for given mode containing columns ???
    :param mode:       string. 'duffing' or 'real'
    :param win_len:    int. for Savitzky Golay filter on data, must be odd
    :param porder:     int. for Savitzky Golay filter on data, polynomial fit
    :param file:       string. file name for real data (not duffing)
    :return: df:       pandas dataframe. the whole data in time.
    """

    df = pd.DataFrame()

    if mode == 'duffing':
        [t, V_R, Q, V_in] = get_data(mode)
        df['T'] = t
        df['V_R'] = V_R
        df['Q'] = Q
        df['V_in'] = V_in

        # dt calculation of time steps
        dt = t[1:] - t[:-1]
        dt = np.append(dt, dt[-1])
        dt[dt == 0] = np.mean(dt)  # do not allow 0 dt
        df['dt'] = dt
    else:  # mode = 'real'
        df = get_data_pd('real2', win_len=win_len, porder=porder, file=file)

        if demean:
            df['V_R'] = df['V_R'] - np.mean(df['V_R'])

        # calculate Q without any weights (hence weight=1)
        Q = calculators.calc_Q_w_weight(df['V_R'].values, df['dt'].values, weight=1)
        if detrend:
            # construct linear fit on Q in time and detrend it from Q. Used only for normalization consts.
            lin_fit = np.polyfit(df['T'].values, Q, 1)
            lin_profile = lin_fit[0] * df['T'].values + lin_fit[1]
            df['Q'] = Q - lin_profile
        else:
            df['Q'] = Q

    return df


def build_df_noQ(mode, win_len=0, porder=0, file=0, demean=0, take_every=1):
    """
    builds Pandas DataFrame for given mode containing columns ???
    :param mode:       string. 'duffing' or 'real'
    :param win_len:    int. for Savitzky Golay filter on data, must be odd
    :param porder:     int. for Savitzky Golay filter on data, polynomial fit
    :param file:       string. file name for real data (not duffing)
    :param demean      int. should measured data be demeaned (1) or not (0) (not duffing)
    :param take_every  int. downsample data for efficiency, only take every take_every data point (not duffing)
    :return: df:       pandas dataframe. the whole data in time.
    """

    df = pd.DataFrame()

    if mode == 'duffing':
        [t, V_R, Q, V_in] = get_data(mode)
        df['T'] = t
        df['V_R'] = V_R
        df['V_in'] = V_in

        # dt calculation of time steps
        dt = t[1:] - t[:-1]
        dt = np.append(dt, dt[-1])
        dt[dt == 0] = np.mean(dt)  # do not allow 0 dt
        df['dt'] = dt

    elif mode == 'period doubling bif':
        df = get_data_pd('period doubling bif', win_len=win_len, porder=porder, file=file, take_every=take_every)
    else:  # mode = 'real'
        df = get_data_pd('real2', win_len=win_len, porder=porder, file=file, take_every=take_every)

        if demean:
            df['V_R'] = df['V_R'] - np.mean(df['V_R'])

    return df


def normalize(t, V_R, V_in):
    """
    normalize function takes the vectors for time t, voltage on resistor V_R and input voltage V_in and normalizes them
    accordingly: t is divided by the period of the input voltage. V_R is divided by the peak to peak voltage of V_R.
    V_in is divided by the peak to peak voltage of V_in.

    :param t: vector of time
    :param V_R: vector of voltage on resistor
    :param V_in: vector of input voltage

    :return: t, V_R, V_in normalized and in dimensionless form
    """

    if any(V_in):
        V_in_pks = find_peaks(V_in)[0]
        T = t[V_in_pks[1]] - t[V_in_pks[0]]
        V_in_p2p = max(V_in) - min(V_in)
    else:
        V_R_pks = find_peaks(V_R)[0]
        T = t[V_R_pks[1]] - t[V_R_pks[0]]
        V_in_p2p = 1
    V_R_p2p = max(V_R) - min(V_R)

    t = t / T
    # t = t / T * np.pi
    # V_R = V_R / (V_R_p2p / np.sqrt(2))
    # V_in = V_in / (V_in_p2p / np.sqrt(2))
    V_R = V_R / (V_R_p2p)
    V_in = V_in / (V_in_p2p)

    return t, V_R, V_in


def to_tensor(df, name_list):
    """
    converts a list of variable names name_list saved in the dataframe df to a list of TF tensors
    :param df:            DataFrame where all the relevant data is
    :param name_list:     list of variable names from df to turn into TF tensors
    :return: list of TF tensors
    """
    lst = []
    for i, name in enumerate(name_list):
        lst.append(tf.convert_to_tensor(df[name].values, dtype=tf.float32))
    return lst


def get_data(dynamics, win_len=0, porder=0, file=0):
    """
    get_data gets V_R, V_in as asked for by user
    :param dynamics: string representing which type of dynamics is used.
                     'real': data on resistor from Roie's lab B - bad data, not recommended
                     'real2': data on resistor from Roie's lab B - good data, recommended
                     'random': dynamics not obeying a PDE
                     'exponent': dV_R/dt = 3/200 * V_R with initial conds. V_R(t=0) = 3
                     'custom': dV_R/dt = 2*t + 1/2 with initial conds. V_R(t=0) = 0
                     'custom2': dV_R/dt = 1/128 * t - 64 * V_in with initial conds. V_R(t=0) = 0
                     'custom3': dV_R/dt = -a^2 * Q with initial conds. V_R(t=0) = a and with a = 1/3
    :param win_len: window length for the Savitzky-Golay filter for real data
    :param porder: order of polynomial for the Savitzky-Golay filter for real and real2
    :param file: string. name of the file to read from for real and real2
    :return: t, V_R, V_in vectors of floats (filtered with Savitzky-Golay for real and real2)
    """

    if dynamics == 'real':
        # V_R
        with open('E:\Work\RLD\data sets\Lab 2 on resistor\V_R.csv', newline='') as csvfile:
            V_R_list = list(csv.reader(csvfile))
            # save just the first array
            V_R_list = copy.copy(V_R_list[0])
            # convert to a readable something
            V_R = np.empty([len(V_R_list)])
            for i, val in enumerate(V_R_list):
                if i == len(V_R_list)-1:
                    V_R[i] = V_R[i-1]
                else:
                    V_R[i] = float(val)

        # V_in
        with open('E:\Work\RLD\data sets\Lab 2 on resistor\V_in.csv', newline='') as csvfile:
            V_in_list = list(csv.reader(csvfile))
            # save just the first array
            V_in_list = copy.copy(V_in_list[0])
            # convert to a readable something
            V_in = np.empty([len(V_in_list)])
            for i, val in enumerate(V_in_list):
                if i == len(V_in_list)-1:
                    V_in[i] = V_in[i-1]
                else:
                    V_in[i] = float(val)

        # # low pass filter from user parameters - not in use
        # V_R = extras.lowPass(V_R, freq, order)
        # V_in = extras.lowPass(V_R, freq, order)

        # Savitzky Golay filter from user parameters - window length and poly-order
        V_R = savgol_filter(V_R, window_length=win_len, polyorder=porder)
        V_in = savgol_filter(V_R, window_length=win_len, polyorder=porder)

        t = np.linspace(0, 100, len(V_R))
        return t, V_R, V_in

    elif dynamics == 'real2':
        with open(file, newline='') as csvfile:
            lst = list(csv.reader(csvfile))
            t = np.empty([len(lst)])
            V_R = np.empty([len(lst)])
            V_in = np.empty([len(lst)])
            for i, val in enumerate(lst):
                if i == len(lst)-1:
                    t[i] = t[i-1]
                    V_in[i] = V_in[i-1]
                    V_R[i] = V_R[i-1]
                else:
                    t[i] = float(val[0])
                    V_in[i] = float(val[1])
                    V_R[i] = float(val[2])

        # # low pass filter from user parameters - not in use
        # V_R = extras.lowPass(V_R, freq, order)
        # V_in = extras.lowPass(V_R, freq, order)

        # Savitzky Golay filter from user parameters - window length and poly-order
        V_R = savgol_filter(V_R, window_length=win_len, polyorder=porder)
        V_in = savgol_filter(V_in, window_length=win_len, polyorder=porder)

        return t, V_R, V_in

    elif dynamics == 'random':
        t = np.linspace(0, 460, 200000)
        V_R = 10*np.random.rand(len(t))
        V_R = V_R - sum(V_R)/len(V_R)
        V_in = 10*np.sin(t)

    elif dynamics == 'exponent':
        t = np.linspace(0, 460, 200000)
        V_R = 3*np.exp(t/200)
        norm_t = t * (8 * np.pi) * 1 / max(t)
        V_in = 1*np.cos(norm_t)+1

    elif dynamics == 'custom':
        t = np.linspace(0, 460, 200000)
        V_R = t**2 + 1/2*t
        V_in = 1*np.cos(t)

    elif dynamics == 'custom2':
        t = np.linspace(0, 460, 200000)
        V_in = 1*np.cos(t)
        V_R = 1/256*t**2 - 64*np.sin(t)

    elif dynamics == 'custom3':
        t = np.linspace(0, 460, 200000)
        a = 1/3
        V_in = 1 / 24 * np.cos(5 * t)
        V_R = a * np.cos(a * t)

    elif dynamics == 'duffing':
        t = np.linspace(0, 460, 200000)
        alpha = 1
        beta = 5
        delta = 0.02
        gamma = 8
        omega = 0.5
        t, V_R, Q, V_in = duffing_dynamics.main()
        # V_in = np.zeros(len(t))
        return t, V_R, Q, V_in

    else:
        return 'wrong dynamics type'

    return t, V_R, V_in


# # NOT IN USE

# def get_df_and_dataset(sizes_for_comb, periods_back, file=None, demean=0):
#     if not demean:
#         print('notice the V_R is not demeaned')
#
#     df = build_df(mode, detrend=0, win_len=window_len, porder=polyorder, file=file, demean=demean)
#     env_V_in = extras.env(df['V_in'], window_len, polyorder - 2)
#     env_V_in = np.polyval(np.polyfit(df['T'], env_V_in, deg=1), df['T'])
#     # arr_for_comb = [np.ones(len(df['V_R'])), df['V_R'], df['Q'], df['V_in']]
#     # arr_for_comb = [np.ones(len(df['V_R'])), df['V_R'], df['Q'], df['V_in'], np.roll(df['V_R'], period),
#     #                 np.roll(df['Q'], period), np.roll(df['V_R'], period * 2), np.roll(df['Q'], period * 2),
#     #                 np.roll(df['V_R'], period * 3), np.roll(df['Q'], period * 3), np.roll(df['V_R'], period * 4),
#     #                 np.roll(df['Q'], period * 4)]
#     arr_for_comb = [np.ones(len(df['V_R'])), df['V_R'], df['V_in'], env_V_in, np.roll(df['V_R'], period),
#                     np.roll(df['V_R'], period * 2), np.roll(df['V_R'], period * 3), np.roll(df['V_R'], period * 4),
#                     np.roll(df['V_R'], period * 5)]
#     dataset_lst = list(combinations_with_replacement(arr_for_comb, 4))
#
#     # dataset_lst = list(combinations_with_replacement([np.ones(len(df['V_R'])), df['V_R'], df['Q'], df['V_in']], 3))
#     dataset_lng = np.transpose(np.array([np.prod(i, axis=0) for i in dataset_lst]))
#     # dataset_lng = np.concatenate((dataset_lng, np.array([np.roll(df['V_R'], period)]).T), axis=1)
#     # dataset_lng = np.concatenate((dataset_lng, np.array([np.roll(df['Q'], period)]).T), axis=1)
#     # dataset_lng = np.concatenate((dataset_lng, np.array([np.roll(df['V_R'], period*2)]).T), axis=1)
#     # dataset_lng = np.concatenate((dataset_lng, np.array([np.roll(df['Q'], period*2)]).T), axis=1)
#     # dataset_lng = np.concatenate((dataset_lng, np.array([np.roll(df['V_R'], period*3)]).T), axis=1)
#     # dataset_lng = np.concatenate((dataset_lng, np.array([np.roll(df['Q'], period*3)]).T), axis=1)
#     # dataset_lng = np.concatenate((dataset_lng, np.array([np.roll(df['V_R'], period*4)]).T), axis=1)
#     # dataset_lng = np.concatenate((dataset_lng, np.array([np.roll(df['Q'], period*4)]).T), axis=1)
#
#     # add V_Rt at the end as result
#     V_Rt = extras.good_derivative(df['V_R'].values, df['dt'].values, order=8)
#     dataset_lng = np.concatenate((dataset_lng, np.array([V_Rt]).T), axis=1)

