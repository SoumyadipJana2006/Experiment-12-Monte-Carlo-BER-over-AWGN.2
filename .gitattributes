# ============================================================
# EXPERIMENT 12 — MONTE CARLO BER OVER AWGN
# BPSK, QPSK, BFSK, 8-PSK, 16-QAM, 64-QAM
#
# Required plots:
# 1. Publication-quality semilog BER curves
# 2. Error count versus transmitted bits
#
# Stopping rule:
#   Stop after 200 errors OR 10^6 transmitted bits
#
# Zero-error trials:
#   BER is NOT set to zero. They are shown as an upper bound:
#   BER < 1/N
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from math import erfc
from pathlib import Path


# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------

EBN0_DB = np.arange(0, 21, 2)      # 0, 2, ..., 20 dB
MAX_ERRORS = 200
MAX_BITS = 10**6
BATCH_BITS = 100_000
ERROR_PLOT_EBN0 = 6                # Eb/N0 for error-count plot

rng = np.random.default_rng(12345)

modulations = [
    "BPSK",
    "QPSK",
    "BFSK",
    "8-PSK",
    "16-QAM",
    "64-QAM"
]

bits_per_symbol = {
    "BPSK": 1,
    "QPSK": 2,
    "BFSK": 1,
    "8-PSK": 3,
    "16-QAM": 4,
    "64-QAM": 6
}


# ------------------------------------------------------------
# Q-FUNCTION
# ------------------------------------------------------------

def Q(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * np.vectorize(erfc)(x / np.sqrt(2))


# ------------------------------------------------------------
# GRAY CODING
# ------------------------------------------------------------

def gray_encode(x):
    return x ^ (x >> 1)


def gray_decode(x):
    x = np.asarray(x, dtype=int).copy()
    shift = 1

    while True:
        shifted = x >> shift

        if not np.any(shifted):
            break

        x ^= shifted
        shift <<= 1

    return x


# ------------------------------------------------------------
# BIT <-> INTEGER CONVERSION
# ------------------------------------------------------------

def bits_to_int(bits, k):
    bits = np.asarray(bits, dtype=np.uint8)
    bits = bits.reshape(-1, k)

    weights = 2 ** np.arange(k - 1, -1, -1)

    return bits.dot(weights)


def int_to_bits(indices, k):
    indices = np.asarray(indices, dtype=np.int64)

    shifts = np.arange(k - 1, -1, -1)

    return ((indices[:, None] >> shifts) & 1).astype(np.uint8).ravel()


# ------------------------------------------------------------
# M-PSK MODULATOR
# ------------------------------------------------------------

def psk_mod(bits, M):

    k = int(np.log2(M))

    indices = bits_to_int(bits, k)

    # Gray-coded symbol index
    gray_indices = gray_encode(indices)

    phase = 2 * np.pi * gray_indices / M

    # Normalized Es = 1
    symbols = np.exp(1j * phase)

    return symbols, k


# ------------------------------------------------------------
# M-PSK DEMODULATOR
# ------------------------------------------------------------

def psk_demod(symbols, M):

    k = int(np.log2(M))

    phase = np.mod(np.angle(symbols), 2 * np.pi)

    gray_indices = np.mod(
        np.rint(phase * M / (2 * np.pi)).astype(int),
        M
    )

    # Build inverse Gray mapping
    inverse_gray = np.empty(M, dtype=int)

    for i in range(M):
        inverse_gray[gray_encode(i)] = i

    indices = inverse_gray[gray_indices]

    return int_to_bits(indices, k)


# ------------------------------------------------------------
# BFSK MODULATION
# ------------------------------------------------------------

def bfsk_mod(bits):

    bits = np.asarray(bits, dtype=np.uint8)

    symbols = np.zeros((len(bits), 2))

    # bit 0 -> first orthogonal basis function
    symbols[bits == 0, 0] = 1

    # bit 1 -> second orthogonal basis function
    symbols[bits == 1, 1] = 1

    return symbols, 1


# ------------------------------------------------------------
# BFSK COHERENT DETECTOR
# ------------------------------------------------------------

def bfsk_demod(received):

    # Compare correlator outputs
    return (received[:, 1] > received[:, 0]).astype(np.uint8)


# ------------------------------------------------------------
# QAM MODULATOR
# ------------------------------------------------------------

def qam_mod(bits, M):

    k = int(np.log2(M))
    L = int(np.sqrt(M))

    bits_axis = k // 2

    indices = bits_to_int(bits, k)

    i_index = indices >> bits_axis
    q_index = indices & ((1 << bits_axis) - 1)

    # Gray mapping on each axis
    i_gray = gray_encode(i_index)
    q_gray = gray_encode(q_index)

    levels = np.arange(-(L - 1), L, 2, dtype=float)

    I = levels[i_gray]
    Qc = levels[q_gray]

    # Normalize average symbol energy to 1
    scale = np.sqrt((2 / 3) * (M - 1))

    symbols = (I + 1j * Qc) / scale

    return symbols, k


# ------------------------------------------------------------
# QAM DEMODULATOR
# ------------------------------------------------------------

def qam_demod(symbols, M):

    k = int(np.log2(M))
    L = int(np.sqrt(M))

    bits_axis = k // 2

    scale = np.sqrt((2 / 3) * (M - 1))

    levels = np.arange(-(L - 1), L, 2, dtype=float)

    # Nearest level detection
    I_index = np.argmin(
        np.abs(
            symbols.real[:, None] * scale -
            levels[None, :]
        ),
        axis=1
    )

    Q_index = np.argmin(
        np.abs(
            symbols.imag[:, None] * scale -
            levels[None, :]
        ),
        axis=1
    )

    # Gray decode
    i_index = gray_decode(I_index)
    q_index = gray_decode(Q_index)

    indices = (i_index << bits_axis) | q_index

    return int_to_bits(indices, k)


# ------------------------------------------------------------
# TRANSMISSION + DETECTION
# ------------------------------------------------------------

def transmit_receive(tx_bits, EbN0_dB, modulation):

    k = bits_per_symbol[modulation]

    # Es = 1
    Eb = 1 / k

    # Convert Eb/N0 from dB to linear
    EbN0 = 10 ** (EbN0_dB / 10)

    # N0 = Eb / (Eb/N0)
    N0 = Eb / EbN0

    # Variance per real dimension = N0/2
    sigma = np.sqrt(N0 / 2)

    # --------------------------------------------------------
    # BFSK
    # --------------------------------------------------------

    if modulation == "BFSK":

        tx, _ = bfsk_mod(tx_bits)

        noise = sigma * rng.standard_normal(tx.shape)

        rx = tx + noise

        detected_bits = bfsk_demod(rx)

    # --------------------------------------------------------
    # BPSK
    # --------------------------------------------------------

    elif modulation == "BPSK":

        tx, _ = psk_mod(tx_bits, 2)

        noise = sigma * (
            rng.standard_normal(tx.shape)
            + 1j * rng.standard_normal(tx.shape)
        )

        rx = tx + noise

        detected_bits = psk_demod(rx, 2)

    # --------------------------------------------------------
    # QPSK
    # --------------------------------------------------------

    elif modulation == "QPSK":

        tx, _ = psk_mod(tx_bits, 4)

        noise = sigma * (
            rng.standard_normal(tx.shape)
            + 1j * rng.standard_normal(tx.shape)
        )

        rx = tx + noise

        detected_bits = psk_demod(rx, 4)

    # --------------------------------------------------------
    # 8-PSK
    # --------------------------------------------------------

    elif modulation == "8-PSK":

        tx, _ = psk_mod(tx_bits, 8)

        noise = sigma * (
            rng.standard_normal(tx.shape)
            + 1j * rng.standard_normal(tx.shape)
        )

        rx = tx + noise

        detected_bits = psk_demod(rx, 8)

    # --------------------------------------------------------
    # QAM
    # --------------------------------------------------------

    elif modulation in ["16-QAM", "64-QAM"]:

        M = 16 if modulation == "16-QAM" else 64

        tx, _ = qam_mod(tx_bits, M)

        noise = sigma * (
            rng.standard_normal(tx.shape)
            + 1j * rng.standard_normal(tx.shape)
        )

        rx = tx + noise

        detected_bits = qam_demod(rx, M)

    else:
        raise ValueError("Unknown modulation")

    return detected_bits


# ------------------------------------------------------------
# MONTE CARLO SIMULATION FOR ONE Eb/N0 POINT
# ------------------------------------------------------------

def simulate_one_point(EbN0_dB, modulation):

    k = bits_per_symbol[modulation]

    errors = 0
    total_bits = 0

    while errors < MAX_ERRORS and total_bits < MAX_BITS:

        remaining = MAX_BITS - total_bits

        nbits = min(BATCH_BITS, remaining)

        # Keep complete symbols
        nbits = (nbits // k) * k

        if nbits == 0:
            break

        # Random transmitted bits
        tx_bits = rng.integers(
            0, 2, nbits, dtype=np.uint8
        )

        # Detection
        rx_bits = transmit_receive(
            tx_bits,
            EbN0_dB,
            modulation
        )

        # Count bit errors
        new_errors = np.count_nonzero(
            tx_bits != rx_bits
        )

        errors += new_errors
        total_bits += nbits

    return errors, total_bits


# ------------------------------------------------------------
# THEORETICAL BER
# ------------------------------------------------------------

def theoretical_BER(modulation, EbN0_dB):

    gamma = 10 ** (np.asarray(EbN0_dB) / 10)

    # BPSK
    if modulation == "BPSK":

        return Q(np.sqrt(2 * gamma))

    # QPSK
    elif modulation == "QPSK":

        return Q(np.sqrt(2 * gamma))

    # Coherent BFSK
    elif modulation == "BFSK":

        return Q(np.sqrt(gamma))

    # 8-PSK approximation
    elif modulation == "8-PSK":

        M = 8
        k = 3

        return (
            2 / k
            * Q(
                np.sqrt(2 * k * gamma)
                * np.sin(np.pi / M)
            )
        )

    # 16-QAM / 64-QAM approximation
    elif modulation in ["16-QAM", "64-QAM"]:

        M = 16 if modulation == "16-QAM" else 64
        k = int(np.log2(M))

        return (
            (4 / k)
            * (1 - 1 / np.sqrt(M))
            * Q(
                np.sqrt(
                    (3 * k / (M - 1)) * gamma
                )
            )
        )


# ------------------------------------------------------------
# RUN BER SIMULATION
# ------------------------------------------------------------

results = {}

print("\n================ MONTE CARLO BER SIMULATION ================\n")

for modulation in modulations:

    print(f"Running {modulation}...")

    simulated_BER = []
    errors_array = []
    bits_array = []

    for snr in EBN0_DB:

        errors, total_bits = simulate_one_point(
            float(snr),
            modulation
        )

        ber = errors / total_bits

        simulated_BER.append(ber)
        errors_array.append(errors)
        bits_array.append(total_bits)

        print(
            f"Eb/N0 = {snr:2d} dB : "
            f"Errors = {errors:4d}, "
            f"Bits = {total_bits:8d}, "
            f"BER = {ber:.3e}"
        )

    simulated_BER = np.array(simulated_BER)
    errors_array = np.array(errors_array)
    bits_array = np.array(bits_array)

    # For zero-error trials:
    # BER < 1/N
    plot_BER = np.where(
        errors_array > 0,
        simulated_BER,
        1 / bits_array
    )

    results[modulation] = {
        "BER": simulated_BER,
        "plot_BER": plot_BER,
        "errors": errors_array,
        "bits": bits_array
    }


# ============================================================
# PLOT 1 — PUBLICATION-QUALITY SEMILOG BER CURVES
# ============================================================

plt.figure(figsize=(10, 6), dpi=200)

markers = ["o", "s", "^", "D", "P", "X"]
linestyles = ["-", "--", "-.", ":", "-", "--"]

snr_smooth = np.linspace(0, 20, 500)

for modulation, marker, linestyle in zip(
        modulations, markers, linestyles):

    result = results[modulation]

    theory = theoretical_BER(
        modulation,
        snr_smooth
    )

    # Theory
    plt.semilogy(
        snr_smooth,
        theory,
        linestyle=linestyle,
        linewidth=2,
        label=f"{modulation} Theory"
    )

    # Simulation points with errors
    valid = result["errors"] > 0

    plt.semilogy(
        EBN0_DB[valid],
        result["BER"][valid],
        marker=marker,
        linestyle="none",
        markersize=7,
        label=f"{modulation} Simulation"
    )

    # Zero-error trials
    zero = result["errors"] == 0

    if np.any(zero):

        plt.semilogy(
            EBN0_DB[zero],
            result["plot_BER"][zero],
            marker="v",
            linestyle="none",
            markerfacecolor="white",
            markersize=7
        )


# Zero-error explanation
plt.semilogy(
    [],
    [],
    marker="v",
    linestyle="none",
    markerfacecolor="white",
    markersize=7,
    label="Zero-error upper bound (1/N)"
)

plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=13)
plt.ylabel("Bit Error Rate (BER)", fontsize=13)

plt.title(
    "Monte Carlo BER Performance over AWGN",
    fontsize=15,
    fontweight="bold"
)

plt.xlim(0, 20)
plt.ylim(1e-6, 1)

plt.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.3
)

plt.legend(
    fontsize=8,
    ncol=2,
    loc="upper right"
)

plt.tight_layout()

plt.savefig(
    "BER_Monte_Carlo_AWGN.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "BER_Monte_Carlo_AWGN.pdf",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 2 — ERROR COUNT VS TRANSMITTED BITS
# ============================================================

def error_progress(modulation, EbN0_dB):

    k = bits_per_symbol[modulation]

    Eb = 1 / k

    EbN0 = 10 ** (EbN0_dB / 10)

    N0 = Eb / EbN0

    sigma = np.sqrt(N0 / 2)

    errors = 0
    total_bits = 0

    bits_history = []
    errors_history = []

    step = 1000

    while errors < MAX_ERRORS and total_bits < MAX_BITS:

        nbits = min(step, MAX_BITS - total_bits)

        nbits = (nbits // k) * k

        if nbits == 0:
            break

        tx_bits = rng.integers(
            0,
            2,
            nbits,
            dtype=np.uint8
        )

        rx_bits = transmit_receive(
            tx_bits,
            EbN0_dB,
            modulation
        )

        wrong_positions = np.flatnonzero(
            tx_bits != rx_bits
        )

        # Stop exactly at the 200th error
        remaining_errors = MAX_ERRORS - errors

        if len(wrong_positions) >= remaining_errors:

            stop_bit = wrong_positions[
                remaining_errors - 1
            ] + 1

            total_bits += stop_bit
            errors = MAX_ERRORS

            bits_history.append(total_bits)
            errors_history.append(errors)

            break

        errors += len(wrong_positions)
        total_bits += nbits

        bits_history.append(total_bits)
        errors_history.append(errors)

    return (
        np.array(bits_history),
        np.array(errors_history)
    )


plt.figure(figsize=(10, 6), dpi=200)

for modulation, marker, linestyle in zip(
        modulations, markers, linestyles):

    x, y = error_progress(
        modulation,
        ERROR_PLOT_EBN0
    )

    plt.plot(
        x,
        y,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.7,
        markersize=4,
        markevery=max(1, len(x) // 8),
        label=modulation
    )


plt.axhline(
    MAX_ERRORS,
    linestyle="--",
    linewidth=1.5,
    label="Stopping criterion = 200 errors"
)

plt.xlabel(
    "Transmitted Bits",
    fontsize=13
)

plt.ylabel(
    "Cumulative Error Count",
    fontsize=13
)

plt.title(
    f"Error Count vs Transmitted Bits "
    f"at $E_b/N_0$ = {ERROR_PLOT_EBN0} dB",
    fontsize=15,
    fontweight="bold"
)

plt.grid(
    True,
    linestyle="--",
    alpha=0.3
)

plt.legend(
    fontsize=9,
    ncol=2
)

plt.tight_layout()

plt.savefig(
    "Error_Count_vs_Transmitted_Bits.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "Error_Count_vs_Transmitted_Bits.pdf",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# MANDATORY VALIDATION
# ============================================================

print("\n================ VALIDATION ================\n")

print("QPSK Gray Mapping:")
print("00 ->   0 degrees")
print("01 ->  90 degrees")
print("11 -> 180 degrees")
print("10 -> 270 degrees")

# BPSK noiseless validation
test_bits = rng.integers(
    0, 2, 10000, dtype=np.uint8
)

tx, _ = psk_mod(test_bits, 2)
detected = psk_demod(tx, 2)

print(
    "\nBPSK noiseless detection:",
    np.array_equal(test_bits, detected)
)

# QPSK noiseless validation
test_bits_qpsk = rng.integers(
    0, 2, 12000, dtype=np.uint8
)

tx, _ = psk_mod(test_bits_qpsk, 4)
detected = psk_demod(tx, 4)

print(
    "QPSK noiseless detection:",
    np.array_equal(test_bits_qpsk, detected)
)

# BFSK noiseless validation
tx, _ = bfsk_mod(test_bits)
detected = bfsk_demod(tx)

print(
    "BFSK noiseless detection:",
    np.array_equal(test_bits, detected)
)

print("\nSimulation completed successfully.")