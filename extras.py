import numpy as np
import math
import copy
import scipy.signal as sgnl
import matplotlib.pyplot as plt

"""
Extra functions used in go_chaos_simple that are not special to go_chaos_simple

Roie Ezraty 2021
"""


def integrate(V, dt):
    return np.cumsum(V) * dt


def good_derivative(V, dt, order=4):
    """
    good_derivative calculates the numerical derivative in 4th or 8th order of a dynamics vector V
    :param V: vector of floats
    :param dt: vector of floats of time differences between measurements of V
    :param order: int. 4th or 8th
    :return: Vt vector of floats
    """

    if order == 4:
        Vt = np.empty([V.shape[0]])
        # first order
        Vt[1] = (V[2]-V[1])/dt[1]
        Vt[0] = (V[1]-V[0])/dt[0]
        Vt[-1] = (V[-1]-V[-2])/dt[-2]
        Vt[-2] = (V[-2]-V[-3])/dt[-3]
        # third order
        Vt[2:-2] = (-V[4:] + 8*V[3:-1] - 8*V[1:-3] + V[0:-4]) / (12 * dt[2:-2])
    elif order == 8:
        Vt = np.empty([V.shape[0]])
        # first order
        Vt[1] = (V[2]-V[1])/dt[1]
        Vt[0] = (V[1]-V[0])/dt[0]
        Vt[-1] = (V[-1]-V[-2])/dt[-2]
        Vt[-2] = (V[-2]-V[-3])/dt[-3]
        # third order
        Vt[2:4] = (-V[4:6] + 8*V[3:5] - 8*V[1:3] + V[0:2]) / (12 * dt[2:4])
        Vt[-4:-2] = (-V[-2:] + 8*V[-3:-1] - 8*V[-5:-3] + V[-6:-4]) / (12 * dt[-4:-2])
        # fifth order
        Vt[4:-4] = (-1/280*V[8:] + 4/105*V[7:-1] - 1/5*V[6:-2] + 4/5*V[5:-3] - 4/5*V[3:-5]
                    + 1/5*V[2:-6] - 4/105*V[1:-7] + 1/280*V[0:-8])/dt[4:-4]
    return Vt


def good_2nd_derivative(V, dt, order=2):
    if order == 2:
        Vtt = np.empty([V.shape[0]])
        Vtt[1:-1] = (V[:-2] - 2 * V[1:-1] + V[2:]) / dt[1:-1] ** 2
        Vtt[0] = Vtt[1]
        Vtt[-1] = Vtt[-2]
    return Vtt


def integrate_FFT(V, f):
    V_I = np.empty([V.shape[0]], dtype=complex)
    V_I[1:] = np.divide(V[1:], 2 * np.pi * np.array([1j]) * f[1:])
    V_I[0] = V_I[1]
    return V_I


def derivative_FFT(V, f):
    f_comp = np.asarray(f) + 0j
    Vt = np.multiply(V, 2 * np.pi * np.array([1j]) * f_comp)
    return Vt


def integrator4(P, coeffs, coeffs_desc, weight):
    """
    integrator4 takes initial conditions from dynamics in P and integrate over the PDE using coeffs.
    the integration:
    V_R(i+1) = V_R(i) + dV_R/dt(i) * dt(i)
    Q(i+1) = weight * Q(i) + V_R(i) * dt with accounting the fact that the Q is detrended so every integration step
                                         has to be detrended accordingly
    :param P: class of variables of the dynamics
    :param coeffs: vector of floats containing the different coefficients
    :param coeffs_desc: vector of lists representing the names of the variables corresponding to the coefficients
    :param weight: float between 0 and 1, for calculation of Q
    :return: P_new.V_R: integrated (theoretical) voltage on resistor V_R
             P_new.Q: integrate (theoretical) charge Q
             and also creating a new class P_new with all the integrated (theoretical) dynamics in it
    """

    length = len(P.V_R)

    # create new class, insert integrated (theoretical) variables in it later
    class P_new:
        V_R = np.empty([length])
        Q = np.empty([length])

        # same as measured data
        V_in = copy.copy(P.V_in)
        V_in2 = copy.copy(P.V_in2)
        V_in3 = copy.copy(P.V_in3)
        ones = copy.copy(P.ones)
        t = copy.copy(P.t)
        detrend_slope = copy.copy(P.detrend_slope)

        V_R2 = np.empty([length])
        V_R3 = np.empty([length])
        Q2 = np.empty([length])
        Q3 = np.empty([length])
        Q4 = np.empty([length])
        Q5 = np.empty([length])
        V_RQ = np.empty([length])
        V_inQ = np.empty([length])
        V_RQ2 = np.empty([length])
        V_inQ2 = np.empty([length])
        V_in2Q = np.empty([length])
        V_R2Q = np.empty([length])
        V_RV_in = np.empty([length])
        V_R2V_in = np.empty([length])
        V_RV_in2 = np.empty([length])

        dt = np.empty([length])
        dt[0:-1] = t[1:] - t[0:-1]
        dt[-1] = dt[-2]

    x = np.empty([len(coeffs)])

    # initial conditions
    P_new.V_R[0] = copy.copy(P.V_R[0])
    P_new.Q[0] = copy.copy(P.Q[0])
    P_new.V_R2[0] = P_new.V_R[0] ** 2
    P_new.V_R3[0] = P_new.V_R[0] ** 3
    P_new.Q2[0] = P_new.Q[0]**2
    P_new.Q3[0] = P_new.Q[0]**3
    P_new.Q4[0] = P_new.Q[0]**4
    P_new.Q5[0] = P_new.Q[0]**5
    P_new.V_RQ[0] = P_new.V_R[0] * P_new.Q[0]
    P_new.V_inQ[0] = P_new.V_in[0] * P_new.Q[0]
    P_new.V_RQ2[0] = P_new.V_R[0] * P_new.Q2[0]
    P_new.V_inQ2[0] = P_new.V_in[0] * P_new.Q2[0]
    P_new.V_in2Q[0] = P_new.V_in2[0] * P_new.Q[0]
    P_new.V_R2Q[0] = P_new.V_R2[0] * P_new.Q[0]
    P_new.V_RV_in[0] = P_new.V_R[0] * P_new.V_in[0]
    P_new.V_R2V_in[0] = P_new.V_R2[0] * P_new.V_in[0]
    P_new.V_RV_in2[0] = P_new.V_R[0] * P_new.V_in2[0]

    # integrate, loop over all indices for the relevant time
    for ind in enumerate(P.V_R[1:]):
        i = ind[0]
        # get data of the right variables from old class P in a vector x for the current time step
        for j, val2 in enumerate(coeffs_desc):
            x[j] = getattr(P_new, val2)[i]

        # the integration step
        P_new.V_R[i+1] = P_new.V_R[i] + np.dot(coeffs, x) * P_new.dt[i]
        if P.Q_mod == 'new':
            P_new.Q[i+1] = weight * P_new.Q[i] + P_new.V_R[i+1] * P_new.dt[i] - P_new.detrend_slope * P_new.dt[i]
        elif P.Q_mod == 'old':
            P_new.Q[i+1] = weight * P_new.Q[i] + P_new.V_R[i+1] * P_new.dt[i]

        # update all other variables in P_new
        P_new.V_R2[i+1] = P_new.V_R[i+1] ** 2
        P_new.V_R3[i+1] = P_new.V_R[i+1] ** 3
        P_new.Q2[i+1] = P_new.Q[i+1]**2
        P_new.Q3[i+1] = P_new.Q[i+1]**3
        P_new.Q4[i+1] = P_new.Q[i+1]**4
        P_new.Q5[i+1] = P_new.Q[i+1]**5
        P_new.V_RQ[i+1] = P_new.V_R[i+1] * P_new.Q[i+1]
        P_new.V_inQ[i+1] = P_new.V_in[i+1] * P_new.Q[i+1]
        P_new.V_RQ2[i+1] = P_new.V_R[i+1] * P_new.Q2[i+1]
        P_new.V_inQ2[i+1] = P_new.V_in[i+1] * P_new.Q2[i+1]
        P_new.V_in2Q[i+1] = P_new.V_in2[i+1] * P_new.Q[i+1]
        P_new.V_R2Q[i+1] = P_new.V_R2[i+1] * P_new.Q[i+1]
        P_new.V_RV_in[i+1] = P_new.V_R[i+1] * P_new.V_in[i+1]
        P_new.V_R2V_in[i+1] = P_new.V_R2[i+1] * P_new.V_in[i+1]
        P_new.V_RV_in2[i+1] = P_new.V_R[i+1] * P_new.V_in2[i+1]

    return P_new.V_R, P_new.Q


def env(signal, window=15, order=3):
    from scipy.signal import hilbert, chirp
    from scipy.signal import savgol_filter

    Hil = hilbert(signal)
    abs_Hil = np.abs(Hil)
    savgol_Hil = savgol_filter(abs_Hil, window, order)
    return savgol_Hil


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1


    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s)
        # pre-sorting of locals min based on relative position with respect to s_mid
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid
        lmax = lmax[s[lmax]>s_mid]


    # global max of dmax-chunks of locals max
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]

    return lmin,lmax


# Low Pass using Butterworth to smooth V_R - probably not in use
def lowPass(data, freq, order=1):
    [b, a] = sgnl.butter(order, freq)
    data_filt = sgnl.lfilter(b, a, data)
    data_filt[0] = data_filt[1]  # correct for first value being close to 0
    return data_filt


def binom(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


def monoExp(x, m, t, b):
    return m * np.exp(-t * x) + b


# # NOT IN USE
#
# def shorten(vecs, N):
#     short_vecs = np.empty([len(vecs), len(vecs[0])-N])
#     for k, vec in enumerate(vecs):
#         short_vecs[k, :] = vec[N:]
#
#     return short_vecs
#
# def movmean(vecs, N):
#     """
#     movmean produces the moving average of multiple vectors
#     :param vecs: 2d array of vectors
#     :param N: window size to do the movmean upon
#     :return: moving_aves: 2d array of averaged vectors
#     """
#
#     moving_aves = np.empty([len(vecs), len(vecs[0])])
#     for k, vec in enumerate(vecs):
#         cumsum = [0]
#         moving_aves_vec = []
#         for i, x in enumerate(vec, 1):
#             cumsum.append(cumsum[i-1] + x)
#             if i >= N:
#                 moving_ave = (cumsum[i] - cumsum[i-N])/N
#                 # can do stuff with moving_ave here
#                 moving_aves_vec.append(moving_ave)
#             else:
#                 moving_aves_vec.append(x)
#         moving_aves[k] = moving_aves_vec
#
#     return moving_aves

# # Same as intergrator2 but accounting for the Q that is calculated with sigmoid
# def integrator3(P, coeffs, coeffs_desc):
#     length = len(P.V_R)
#
#     class P_new:
#         V_R = np.empty([length])
#         V_in = copy.copy(P.V_in)
#         V_in2 = copy.copy(P.V_in2)
#         ones = copy.copy(P.ones)
#         t = copy.copy(P.t)
#         Q = np.empty([length])
#
#         V_R2 = np.empty([length])
#         Q2 = np.empty([length])
#         Q3 = np.empty([length])
#         Q4 = np.empty([length])
#         Q5 = np.empty([length])
#         V_RQ = np.empty([length])
#         V_inQ = np.empty([length])
#         V_RQ2 = np.empty([length])
#         V_inQ2 = np.empty([length])
#         V_in2Q = np.empty([length])
#         V_R2Q = np.empty([length])
#         V_RV_in = np.empty([length])
#         V_R2V_in = np.empty([length])
#         V_RV_in2 = np.empty([length])
#
#         dt = np.empty([length])
#         dt[0:-1] = t[1:] - t[0:-1]
#         dt[-1] = dt[-2]
#
#         V_R_tot = np.concatenate((P.V_R_memory, V_R))
#
#     x = np.empty([len(coeffs)])
#
#     P_new.V_R[0] = copy.copy(P.V_R[0])
#     P_new.Q[0] = copy.copy(P.Q[0])
#     P_new.V_R2[0] = P_new.V_R[0] ** 2
#     P_new.Q2[0] = P_new.Q[0]**2
#     P_new.Q3[0] = P_new.Q[0]**3
#     P_new.Q4[0] = P_new.Q[0]**4
#     P_new.Q5[0] = P_new.Q[0]**5
#     P_new.V_RQ[0] = P_new.V_R[0] * P_new.Q[0]
#     P_new.V_inQ[0] = P_new.V_in[0] * P_new.Q[0]
#     P_new.V_RQ2[0] = P_new.V_R[0] * P_new.Q2[0]
#     P_new.V_inQ2[0] = P_new.V_in[0] * P_new.Q2[0]
#     P_new.V_in2Q[0] = P_new.V_in2[0] * P_new.Q[0]
#     P_new.V_R2Q[0] = P_new.V_R2[0] * P_new.Q[0]
#     P_new.V_RV_in[0] = P_new.V_R[0] * P_new.V_in[0]
#     P_new.V_R2V_in[0] = P_new.V_R2[0] * P_new.V_in[0]
#     P_new.V_RV_in2[0] = P_new.V_R[0] * P_new.V_in2[0]
#
#     P_new.V_R_tot[length] = P_new.V_R[0]
#
#     for ind in enumerate(P.V_R[1:]):
#         i = ind[0]
#         # if i == 500:
#         #     y = 6
#         for j, val2 in enumerate(coeffs_desc):
#             # x[j] = P_new.__dict__.get(val2)[i]
#             x[j] = getattr(P_new, val2)[i]
#         # Q = P.Q[i-1]
#         # Vin = P.V_in[i-1]
#         # VR = V_R[i-1]
#
#         P_new.V_R[i+1] = P_new.V_R[i] + np.dot(coeffs, x) * P_new.dt[i]
#         P_new.V_R_tot[length+i+1] = P_new.V_R[i+1]
#
#         P_new.Q[i+1] = np.convolve(P_new.V_R_tot, P.sigmoid)[length+i] * P_new.dt[i]
#
#         P_new.V_R2[i+1] = P_new.V_R[i+1] ** 2
#         P_new.Q2[i+1] = P_new.Q[i+1]**2
#         P_new.Q3[i+1] = P_new.Q[i+1]**3
#         P_new.Q4[i+1] = P_new.Q[i+1]**4
#         P_new.Q5[i+1] = P_new.Q[i+1]**5
#         P_new.V_RQ[i+1] = P_new.V_R[i+1] * P_new.Q[i+1]
#         P_new.V_inQ[i+1] = P_new.V_in[i+1] * P_new.Q[i+1]
#         P_new.V_RQ2[i+1] = P_new.V_R[i+1] * P_new.Q2[i+1]
#         P_new.V_inQ2[i+1] = P_new.V_in[i+1] * P_new.Q2[i+1]
#         P_new.V_in2Q[i+1] = P_new.V_in2[i+1] * P_new.Q[i+1]
#         P_new.V_R2Q[i+1] = P_new.V_R2[i+1] * P_new.Q[i+1]
#         P_new.V_RV_in[i+1] = P_new.V_R[i+1] * P_new.V_in[i+1]
#         P_new.V_R2V_in[i+1] = P_new.V_R2[i+1] * P_new.V_in[i+1]
#         P_new.V_RV_in2[i+1] = P_new.V_R[i+1] * P_new.V_in2[i+1]
#
#     return P_new.V_R, P_new.Q

# def integratorRK4(P, coeffs, coeffs_desc):
#
#     class P_new:
#         V_R = np.empty([len(P.V_R)])
#         V_in = copy.copy(P.V_in)
#         V_in2 = copy.copy(P.V_in2)
#         ones = copy.copy(P.ones)
#         t = copy.copy(P.t)
#         Q = np.empty([len(P.V_R)])
#         V_R2 = np.empty([len(P.V_R)])
#         Q2 = np.empty([len(P.V_R)])
#         Q3 = np.empty([len(P.V_R)])
#         Q4 = np.empty([len(P.V_R)])
#         Q5 = np.empty([len(P.V_R)])
#         V_RQ = np.empty([len(P.V_R)])
#         V_inQ = np.empty([len(P.V_R)])
#         V_RQ2 = np.empty([len(P.V_R)])
#         V_inQ2 = np.empty([len(P.V_R)])
#         V_in2Q = np.empty([len(P.V_R)])
#         V_R2Q = np.empty([len(P.V_R)])
#         V_RV_in = np.empty([len(P.V_R)])
#         V_R2V_in = np.empty([len(P.V_R)])
#         V_RV_in2 = np.empty([len(P.V_R)])
#
#         dt = np.empty([len(P.V_R)])
#         dt[0:-1] = t[1:] - t[0:-1]
#         dt[-1] = dt[-2]
#
#     x1 = np.empty([len(coeffs)])
#     x2 = np.empty([len(coeffs)])
#     x3 = np.empty([len(coeffs)])
#     x4 = np.empty([len(coeffs)])
#
#     for i, val in enumerate(P.V_R):
#         if i == 0:
#             P_new.V_R[i] = copy.copy(P.V_R[0])
#             P_new.Q[i] = copy.copy(P.Q[0])
#         else:
#             for j, val2 in enumerate(coeffs_desc):
#                 x1[j] = getattr(P_new, val2)[i-1]
#             k1 = np.dot(coeffs, x1) * P_new.dt[i-1]
#             update_P_new(P_new, i-1, k1/2)
#
#             for j, val2 in enumerate(coeffs_desc):
#                 if val2 == 'ones':
#                     x2[j] = getattr(P_new, val2)[i-1]
#                 elif val2 == 't':
#                     x2[j] = getattr(P_new, val2)[i-1] + P_new.dt[i-1] / 2
#                 else:
#                     x2[j] = getattr(P_new, val2)[i-1]
#             k2 = np.dot(coeffs, x2) * P_new.dt[i-1]
#             update_P_new(P_new, i-1, (k2-k1)/2)
#
#             for j, val2 in enumerate(coeffs_desc):
#                 if val2 == 'ones':
#                     x3[j] = getattr(P_new, val2)[i-1]
#                 elif val2 == 't':
#                     x3[j] = getattr(P_new, val2)[i-1] + P_new.dt[i-1] / 2
#                 else:
#                     x3[j] = getattr(P_new, val2)[i-1]
#             k3 = np.dot(coeffs, x3) * P_new.dt[i-1]
#             update_P_new(P_new, i-1, k3-k2/2)
#
#             for j, val2 in enumerate(coeffs_desc):
#                 if val2 == 'ones':
#                     x4[j] = getattr(P_new, val2)[i-1]
#                 elif val2 == 't':
#                     x4[j] = getattr(P_new, val2)[i-1] + P_new.dt[i-1]
#                 else:
#                     x4[j] = getattr(P_new, val2)[i-1]
#             k4 = np.dot(coeffs, x4) * P_new.dt[i-1]
#             update_P_new(P_new, i-1, -k3)
#             P_new.V_R[i] = P_new.V_R[i-1] + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
#
#             P_new.Q[i] = P_new.Q[i-1] + P_new.V_R[i] * P_new.dt[i-1]
#
#             update_P_new(P_new, i-1, 0)
#
#     return P_new.V_R, P_new.Q
#
# def update_P_new(P_new, i, deltaK):
#
#     P_new.V_R[i] = P_new.V_R[i] + deltaK
#     P_new.Q[i] = P_new.Q[i] + deltaK * P_new.dt[i-1]
#     P_new.V_R2[i] = P_new.V_R[i] ** 2
#     P_new.Q2[i] = P_new.Q[i]**2
#     P_new.Q3[i] = P_new.Q[i]**3
#     P_new.Q4[i] = P_new.Q[i]**4
#     P_new.Q5[i] = P_new.Q[i]**5
#     P_new.V_RQ[i] = P_new.V_R[i] * P_new.Q[i]
#     P_new.V_inQ[i] = P_new.V_in[i] * P_new.Q[i]
#     P_new.V_RQ2[i] = P_new.V_R[i] * P_new.Q2[i]
#     P_new.V_inQ2[i] = P_new.V_in[i] * P_new.Q2[i]
#     P_new.V_in2Q[i] = P_new.V_in2[i] * P_new.Q[i]
#     P_new.V_R2Q[i] = P_new.V_R2[i] * P_new.Q[i]
#     P_new.V_RV_in[i] = P_new.V_R[i] * P_new.V_in[i]
#     P_new.V_R2V_in[i] = P_new.V_R2[i] * P_new.V_in[i]
#     P_new.V_RV_in2[i] = P_new.V_R[i] * P_new.V_in2[i]

# def integrator2(P, coeffs, coeffs_desc):
#     """
#     integrator2 takes initial conditions
#     :param P:
#     :param coeffs:
#     :param coeffs_desc:
#     :return:
#     """
#
#     # length = len(P.V_R)
#     # V_R = np.empty([length])
#     # V_in = copy.copy(P.V_in)
#     # V_in2 = copy.copy(P.V_in2)
#     # ones = copy.copy(P.ones)
#     # t = copy.copy(P.t)
#     # Q = np.empty([length])
#     # V_R2 = np.empty([length])
#     # Q2 = np.empty([length])
#     # Q3 = np.empty([length])
#     # Q4 = np.empty([length])
#     # Q5 = np.empty([length])
#     # V_RQ = np.empty([length])
#     # V_inQ = np.empty([length])
#     # V_RQ2 = np.empty([length])
#     # V_inQ2 = np.empty([length])
#     # V_in2Q = np.empty([length])
#     # V_R2Q = np.empty([length])
#     # V_RV_in = np.empty([length])
#     # V_R2V_in = np.empty([length])
#     # V_RV_in2 = np.empty([length])
#     #
#     # dt = np.empty([length])
#     # dt[0:-1] = t[1:] - t[0:-1]
#     # dt[-1] = dt[-2]
#     #
#     # x = np.empty([len(coeffs)])
#     #
#     # V_R[0] = copy.copy(P.V_R[0])
#     # Q[0] = copy.copy(P.Q[0])
#     # V_R2[0] = V_R2[0] ** 2
#     # V_R2[0] = V_R[0] ** 2
#     # Q2[0] = Q[0]**2
#     # Q3[0] = Q[0]**3
#     # Q4[0] = Q[0]**4
#     # Q5[0] = Q[0]**5
#     # V_RQ[0] = V_R[0] * Q[0]
#     # V_inQ[0] = V_in[0] * Q[0]
#     # V_RQ2[0] = V_R[0] * Q2[0]
#     # V_inQ2[0] = V_in[0] * Q2[0]
#     # V_in2Q[0] = V_in2[0] * Q[0]
#     # V_R2Q[0] = V_R2[0] * Q[0]
#     # V_RV_in[0] = V_R[0] * V_in[0]
#     # V_R2V_in[0] = V_R2[0] * V_in[0]
#     # V_RV_in2[0] = V_R[0] * V_in2[0]
#     #
#     # for i, val in enumerate(P.V_R[1:]):
#     #     for j, val2 in enumerate(coeffs_desc):
#     #         x[j] = vars().get(val2)[i]
#     #     V_R[i+1] = V_R[i] + np.dot(coeffs, x) * dt[i]
#     #
#     #     Q[i+1] = Q[i] + V_R[i+1] * dt[i]
#     #
#     #     V_R2[i+1] = V_R[i+1] ** 2
#     #     Q2[i+1] = Q[i+1]**2
#     #     Q3[i+1] = Q[i+1]**3
#     #     Q4[i+1] = Q[i+1]**4
#     #     Q5[i+1] = Q[i+1]**5
#     #     V_RQ[i+1] = V_R[i+1] * Q[i+1]
#     #     V_inQ[i+1] = V_in[i+1] * Q[i+1]
#     #     V_RQ2[i+1] = V_R[i+1] * Q2[i+1]
#     #     V_inQ2[i+1] = V_in[i+1] * Q2[i+1]
#     #     V_in2Q[i+1] = V_in2[i+1] * Q[i+1]
#     #     V_R2Q[i+1] = V_R2[i+1] * Q[i+1]
#     #     V_RV_in[i+1] = V_R[i+1] * V_in[i+1]
#     #     V_R2V_in[i+1] = V_R2[i+1] * V_in[i+1]
#     #     V_RV_in2[i+1] = V_R[i+1] * V_in2[i+1]
#     #
#     # return V_R, Q
#
#     length = len(P.V_R)
#
#     class P_new:
#         V_R = np.empty([length])
#         V_in = copy.copy(P.V_in)
#         V_in2 = copy.copy(P.V_in2)
#         ones = copy.copy(P.ones)
#         t = copy.copy(P.t)
#         Q = np.empty([length])
#
#         V_R2 = np.empty([length])
#         Q2 = np.empty([length])
#         Q3 = np.empty([length])
#         Q4 = np.empty([length])
#         Q5 = np.empty([length])
#         V_RQ = np.empty([length])
#         V_inQ = np.empty([length])
#         V_RQ2 = np.empty([length])
#         V_inQ2 = np.empty([length])
#         V_in2Q = np.empty([length])
#         V_R2Q = np.empty([length])
#         V_RV_in = np.empty([length])
#         V_R2V_in = np.empty([length])
#         V_RV_in2 = np.empty([length])
#
#         dt = np.empty([length])
#         dt[0:-1] = t[1:] - t[0:-1]
#         dt[-1] = dt[-2]
#
#     x = np.empty([len(coeffs)])
#
#     P_new.V_R[0] = copy.copy(P.V_R[0])
#     P_new.Q[0] = copy.copy(P.Q[0])
#     P_new.V_R2[0] = P_new.V_R[0] ** 2
#     P_new.Q2[0] = P_new.Q[0]**2
#     P_new.Q3[0] = P_new.Q[0]**3
#     P_new.Q4[0] = P_new.Q[0]**4
#     P_new.Q5[0] = P_new.Q[0]**5
#     P_new.V_RQ[0] = P_new.V_R[0] * P_new.Q[0]
#     P_new.V_inQ[0] = P_new.V_in[0] * P_new.Q[0]
#     P_new.V_RQ2[0] = P_new.V_R[0] * P_new.Q2[0]
#     P_new.V_inQ2[0] = P_new.V_in[0] * P_new.Q2[0]
#     P_new.V_in2Q[0] = P_new.V_in2[0] * P_new.Q[0]
#     P_new.V_R2Q[0] = P_new.V_R2[0] * P_new.Q[0]
#     P_new.V_RV_in[0] = P_new.V_R[0] * P_new.V_in[0]
#     P_new.V_R2V_in[0] = P_new.V_R2[0] * P_new.V_in[0]
#     P_new.V_RV_in2[0] = P_new.V_R[0] * P_new.V_in2[0]
#
#     for ind in enumerate(P.V_R[1:]):
#         i = ind[0]
#         # if i == 500:
#         #     y = 6
#         for j, val2 in enumerate(coeffs_desc):
#             # x[j] = P_new.__dict__.get(val2)[i]
#             x[j] = getattr(P_new, val2)[i]
#         # Q = P.Q[i-1]
#         # Vin = P.V_in[i-1]
#         # VR = V_R[i-1]
#
#         P_new.V_R[i+1] = P_new.V_R[i] + np.dot(coeffs, x) * P_new.dt[i]
#
#         P_new.Q[i+1] = P_new.Q[i] + P_new.V_R[i+1] * P_new.dt[i]
#
#         P_new.V_R2[i+1] = P_new.V_R[i+1] ** 2
#         P_new.Q2[i+1] = P_new.Q[i+1]**2
#         P_new.Q3[i+1] = P_new.Q[i+1]**3
#         P_new.Q4[i+1] = P_new.Q[i+1]**4
#         P_new.Q5[i+1] = P_new.Q[i+1]**5
#         P_new.V_RQ[i+1] = P_new.V_R[i+1] * P_new.Q[i+1]
#         P_new.V_inQ[i+1] = P_new.V_in[i+1] * P_new.Q[i+1]
#         P_new.V_RQ2[i+1] = P_new.V_R[i+1] * P_new.Q2[i+1]
#         P_new.V_inQ2[i+1] = P_new.V_in[i+1] * P_new.Q2[i+1]
#         P_new.V_in2Q[i+1] = P_new.V_in2[i+1] * P_new.Q[i+1]
#         P_new.V_R2Q[i+1] = P_new.V_R2[i+1] * P_new.Q[i+1]
#         P_new.V_RV_in[i+1] = P_new.V_R[i+1] * P_new.V_in[i+1]
#         P_new.V_R2V_in[i+1] = P_new.V_R2[i+1] * P_new.V_in[i+1]
#         P_new.V_RV_in2[i+1] = P_new.V_R[i+1] * P_new.V_in2[i+1]
#
#     return P_new.V_R, P_new.Q

# def integrator(V_R, V_in, w, Q0, dt, degree=1):
#     ## the structure of w is
#     #
#     # ones
#     # Q
#     # Q**2
#     # Q**3
#     # V_in
#     # V_R
#     # V_in*Q
#     # V_R*Q
#     # V_in*Q**2
#     # V_R*Q**2
#     # V_in*Q**3
#     # V_R*Q**3
#
#     U_vec = np.empty([len(V_R), 1])
#     Q_vec = np.empty([len(V_R), 1])
#
#     if degree == 1:
#         for i, val in enumerate(V_R):
#             if i == 0:
#                 U_vec[i] = copy.copy(V_R[0])
#                 Q_vec[i] = copy.copy(Q0)
#             else:
#                 Q = Q_vec[i-1]
#                 Vin = V_in[i-1]
#                 VR = U_vec[i-1]
#                 U_vec[i] = U_vec[i-1] + \
#                            (w[0] + w[1]*Q + w[2]*Q**2 + w[3]*Q**3 + w[4]*Vin + w[5]*VR
#                             + w[6]*Vin*Q + w[7]*VR*Q + w[8]*Vin*Q**2 + w[9]*VR*Q**2
#                             + w[10]*Vin*Q**3 + w[11]*VR*Q**3) * dt
#
#                 Q_vec[i] = Q_vec[i-1] + U_vec[i] * dt
#
#     elif degree == 2:
#         for i, val in enumerate(V_R):
#             if i == 0:
#                 U_vec[i] = copy.copy(V_R[0])
#                 Q_vec[i] = copy.copy(Q0)
#             else:
#                 Q = Q_vec[i-1]
#                 Vin = V_in[i-1]
#                 VR = U_vec[i-1]
#                 U_vec[i] = U_vec[i-1] + \
#                            (w[0] + w[1]*Q + w[2]*Q**2 + w[3]*Q**3 + w[4]*Vin + w[5]*VR
#                             + w[6]*Vin**2 + w[7]*VR*Vin + w[8]*VR**2 + w[9]*Vin*Q
#                             + w[10]*VR*Q + w[11]*Vin**2*Q + w[12]*VR*Vin + w[8]*Vin*Q**2
#                             + w[9]*VR*Q**2 + w[10]*Vin*Q**3 + w[11]*VR*Q**3 + w[14]*Vin*Q**2
#                             + w[15]*VR*Q**2 + w[16]*Vin**2*Q**2 + w[17]*VR*Vin*Q**2
#                             + w[18]*VR**2*Q**2 + w[19]*Vin*Q**3 + w[20]*VR*Q**3 + w[21]*Vin**2*Q**3
#                             + w[22]*VR*Vin*Q**3 + w[23]*VR**2*Q**3) * dt
#
#                 Q_vec[i] = Q_vec[i-1] + U_vec[i] * dt
#     return U_vec

# def normalize(vecs, normal_type='mean'):
#     lst = [0]
#     if normal_type == 'none':
#         return vecs
#     else:
#         if normal_type == 'mean':
#             for i, val in enumerate(vecs):
#                 lst.append(val / np.mean(val))
#         elif normal_type == 'std':
#             for i, val in enumerate(vecs):
#                 lst.append(val / np.std(val))
#         return lst[1:]
