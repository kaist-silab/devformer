import numpy as np

from src.problems.dpp.simulator import decap_placement, initial_impedance


DEFAULT_DIR = "data/dpp/simulator/"


def model_1(z_initial, z_final, filename=DEFAULT_DIR + "freq_201.npy"):
    freq_pts = 201
    impedance_gap = np.zeros(freq_pts)

    with open(filename, "rb") as f:
        freq = np.load(f)

    freq_point = 2e9
    min = 0.32
    grad = 0.16
    target_impedance = np.zeros(np.shape(freq))
    idx0 = np.argwhere(freq < freq_point)
    idx1 = np.argwhere(freq >= freq_point)
    target_impedance[idx0] = min
    target_impedance[idx1] = grad * 1e-9 * freq[idx1]

    penalty = 1
    reward = 0

    for i in range(freq_pts):
        if z_final[i] > target_impedance[i]:
            impedance_gap[i] = (z_final[i] - target_impedance[i]) * penalty
        else:
            impedance_gap[i] = 0
        # impedance_gap[i]=target_impedance[i]-z_final[i]

        reward = reward - (impedance_gap[i] / (434 * penalty))
    return reward


def model_2(z_initial, z_final, filename=DEFAULT_DIR + "freq_201.npy"):
    freq_pts = 201
    impedance_gap = np.zeros(freq_pts)

    with open(filename, "rb") as f:
        freq = np.load(f)

    freq_point = 2e9
    reward = 0

    for i in range(freq_pts):
        impedance_gap[i] = z_initial[i] - z_final[i]
        reward = reward + impedance_gap[i]
    reward = reward / 10
    return reward


def model_3(z_initial, z_final, filename=DEFAULT_DIR + "freq_201.npy"):
    freq_pts = 201
    impedance_gap = np.zeros(freq_pts)
    with open(filename, "rb") as f:
        freq = np.load(f)

    freq_point = 2e9
    reward = 0

    for i in range(freq_pts):
        impedance_gap[i] = z_initial[i] - z_final[i]

        if freq[i] < freq_point:
            reward = reward + (impedance_gap[i] * 1.5)

        else:
            reward = reward + impedance_gap[i]
    reward = reward / 10
    return reward


def model_4(z_initial, z_final, filename=DEFAULT_DIR + "freq_201.npy"):
    freq_pts = 201
    impedance_gap = np.zeros(freq_pts)

    with open(filename, "rb") as f:
        freq = np.load(f)

    freq_point = 2e9
    reward = 0

    for i in range(freq_pts):
        impedance_gap[i] = z_initial[i] - z_final[i]

        if freq[i] < freq_point:
            if impedance_gap[i] > 0:
                reward = reward + (impedance_gap[i] * 1.5)
            else:
                reward = reward + (impedance_gap[i] * 3)
        else:
            if impedance_gap[i] > 0:
                reward = reward + impedance_gap[i]
            else:
                reward = reward + (impedance_gap[i] * 3)
    reward = reward / 10
    return reward


def model_5(z_initial, z_final, filename=DEFAULT_DIR + "freq_201.npy"):
    freq_pts = 201
    impedance_gap = np.zeros(freq_pts)

    with open(filename, "rb") as f:
        freq = np.load(f)

    freq_point = 2e9
    reward = 0

    for i in range(freq_pts):
        impedance_gap[i] = z_initial[i] - z_final[i]
        reward = reward + (impedance_gap[i] * 1000000000 / freq[i])
    reward = reward / 10
    return reward


def model_6(z_initial, z_final):
    freq_pts = 201
    impedance_gap = np.zeros(freq_pts)
    target_impedance = 0.6 * np.ones(freq_pts)
    reward = 0
    penalty = 1

    for i in range(freq_pts):  # NOTE: 0.013 sec
        if z_final[i] > target_impedance[i]:  # NOTE: size(434)
            impedance_gap[i] = (z_final[i] - target_impedance[i]) * penalty
        else:
            impedance_gap[i] = 0
            # impedance_gap[i]=target_impedance[i]-z_final[i]
        reward = reward - (
            impedance_gap[i] / (434 * penalty)
        )  # TODO: Using torch.mean()

    return reward


def reward_gen(probe, pi, model, filename=DEFAULT_DIR + "10x10_pkg_chip.npy"):
    # n=15
    # m=15
    # with open ('problems/decap/15x15_pkg_chip.npy', 'rb') as f:
    #     raw_pdn=np.load(f)

    n = 10
    m = 10
    with open(filename, "rb") as f:
        raw_pdn = np.load(f)

    z_initial = initial_impedance(n, m, raw_pdn, probe)
    z_initial = np.abs(z_initial)

    pi = pi.astype(int)
    z_final = decap_placement(n, m, raw_pdn, pi, probe, 201)
    z_final = np.abs(z_final)

    if model == 1:
        reward = model_1(z_initial, z_final)

    elif model == 2:
        reward = model_2(z_initial, z_final)

    elif model == 3:
        reward = model_3(z_initial, z_final)

    elif model == 4:
        reward = model_4(z_initial, z_final)

    elif model == 5:
        reward = model_5(z_initial, z_final)

    elif model == 6:
        reward = model_6(z_initial, z_final)

    return reward
