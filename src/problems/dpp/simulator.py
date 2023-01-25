import numpy as np
from numpy.linalg import inv


DEFAULT_DIR = "data/dpp/simulator/"


def decap_placement(
    n,
    m,
    raw_pdn,
    pi,
    probing_port,
    freq_pts,
    fpath=DEFAULT_DIR + "01nF_decap.npy",
):
    num_decap = np.size(pi)
    probe = probing_port
    z1 = raw_pdn

    with open(fpath, "rb") as f:
        decap = np.load(f)
    decap = decap.reshape(-1)
    z2 = np.zeros((freq_pts, num_decap, num_decap))

    qIndx = []
    for i in range(num_decap):
        z2[:, i, i] = np.abs(decap)
        qIndx.append(i)
    pIndx = pi.astype(int)

    # pIndx : index of ports in z1 for connecting
    # qIndx : index of ports in z2 for connecting

    aIndx = np.arange(len(z1[0]))

    aIndx = np.delete(aIndx, pIndx)

    z1aa = z1[:, aIndx, :][:, :, aIndx]
    z1ap = z1[:, aIndx, :][:, :, pIndx]
    z1pa = z1[:, pIndx, :][:, :, aIndx]
    z1pp = z1[:, pIndx, :][:, :, pIndx]
    z2qq = z2[:, qIndx, :][:, :, qIndx]

    zout = z1aa - np.matmul(np.matmul(z1ap, inv(z1pp + z2qq)), z1pa)

    for i in range(n * m):
        if i in pi:

            if i < probing_port:
                probe = probe - 1

    zout = zout[:, probe, probe]
    return zout


def decap_model(z_initial, z_final, N_freq, fpath=DEFAULT_DIR + "freq_201.npy"):

    impedance_gap = np.zeros(N_freq)

    with open(fpath, "rb") as f:
        freq = np.load(f)

    reward = 0
    for i in range(N_freq):
        impedance_gap[i] = z_initial[i] - z_final[i]
        reward = reward + (impedance_gap[i] * 1000000000 / freq[i])
    reward = reward / 10
    return reward


def initial_impedance(n, m, raw_pdn, probe):

    zout = raw_pdn[:, probe, probe]

    return zout


def decap_sim(
    probe,
    solution,
    keepout=None,
    N=10,
    N_freq=201,
    fpath=DEFAULT_DIR + "10x10_pkg_chip.npy",
):

    with open(fpath, "rb") as f:
        raw_pdn = np.load(f)
    solution = np.array(solution)

    assert len(solution) == len(
        np.unique(solution)
    ), "An Element of Decap Sequence must be Unique"

    if keepout is not None:
        keepout = np.array(keepout)
        intersect = np.intersect1d(solution, keepout)
        assert len(intersect) == 0, "Decap must be not placed at the keepout region"

    z_initial = initial_impedance(N, N, raw_pdn, probe)
    z_initial = np.abs(z_initial)
    z_final = decap_placement(N, N, raw_pdn, solution, probe, N_freq)
    z_final = np.abs(z_final)
    reward = decap_model(z_initial, z_final, N_freq)

    return reward
