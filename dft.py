import numpy as np
import matplotlib.pyplot as plt
import math


# Computes the magnitude of each complex number in the array: |z| = sqrt(a^2 + b^2)
# Equivalent to np.abs(values), implemented from scratch for learning purposes.
def complex_absolute_value(values: np.ndarray) -> np.ndarray:
    result = np.array([0.0] * values.size)
    for i in range(values.size):
        result[i] = math.sqrt(x=values[i].real**2 + values[i].imag**2)
    return result


# Computes the Discrete Fourier Transform (DFT) of the input signal.
# Implemented from scratch as an O(N^2) algorithm for learning purposes;
# np.fft.fft() provides an optimized O(N log N) equivalent.
#
#   Continuous Fourier Transform:  X(F) = ∫[-∞,∞] x(t) * e^(-j*2*π*F*t) * dt
#   Discrete Fourier Transform:    Xk   = SUM[n=0..N-1] xn * e^((-j*2*π*k*n)/N)
#      k/N ~ F  (frequency index maps to frequency)
#      n   ~ t  (sample index maps to time)
def discrete_fourier_transform(values: np.ndarray) -> np.ndarray:
    num_samples = values.size
    neg_two_pi_over_n = -math.pi * 2 / num_samples
    result = np.array([0 + 0j] * num_samples)  # output array of complex numbers

    for k in range(num_samples):
        sum_real = 0.0
        sum_imag = 0.0
        for n in range(num_samples):
            exp_value = neg_two_pi_over_n * k * n
            # Euler's formula: e^(j*x) = cos(x) + j*sin(x)
            sum_real += values[n] * math.cos(x=exp_value)
            sum_imag += values[n] * math.sin(x=exp_value)
        result[k] = complex(real=sum_real, imag=sum_imag)

    return result


def run_dft_with_graphs() -> None:
    # --- Input signal ---
    # Composite signal made of four cosine waves at 1, 7, 13, and 15 Hz,
    # each phase-shifted by π/2 (1.571 rad).
    time_span = np.linspace(start=0, stop=0.875 * 5, num=8 * 5 * 32)  # 0 to 4.375 seconds, 1280 samples

    values = (
          np.cos(     2 * np.pi * time_span - 1.571)   #  1 Hz
        + np.cos( 7 * 2 * np.pi * time_span - 1.571)   #  7 Hz
        + np.cos(13 * 2 * np.pi * time_span - 1.571)   # 13 Hz
        + np.cos(15 * 2 * np.pi * time_span - 1.571)   # 15 Hz
    )

    # --- Frequency axis ---
    num_samples   = values.size
    step_interval = time_span[1] - time_span[0]  # time between consecutive samples (seconds)
    sample_rate   = 1 / step_interval             # sampling rate (Hz)

    # Frequency bins from 0 Hz to the sample rate, evenly spaced by sample_rate / num_samples
    frequencies        = np.linspace(start=0, stop=sample_rate, num=num_samples)
    usable_frequencies = frequencies[0 : num_samples // 2]  # Nyquist cutoff: only the lower half is meaningful

    # --- Discrete Fourier Transform ---
    dft_result = discrete_fourier_transform(values=values)
    # np.fft.fft(values)  # numpy O(N log N) equivalent

    # --- Magnitude: |Xk| = sqrt(a^2 + b^2) ---
    dft_magnitude = complex_absolute_value(values=dft_result)
    # np.abs(dft_result)  # numpy equivalent

    # Normalize amplitude over the usable (lower) half of the spectrum.
    # Discarding the upper half halves the total energy, so multiply by 2 to restore it,
    # then divide by N to scale back to the original signal amplitude.
    # e.g. a 1 Hz tone with N=8 yields raw magnitude ~3.9997
    #      3.9997 * 2 / 8 = 0.9999 ≈ 1.0
    amplitude = dft_magnitude[0 : num_samples // 2] * 2 / num_samples

    # --- Plot 1: time-domain signal ---
    plt.title(label="Time-domain signal")
    plt.xlabel(xlabel="Time [s]")
    plt.ylabel(ylabel="Amplitude")
    plt.plot(time_span, values)
    plt.show()

    # --- Plot 2: frequency-domain magnitude spectrum ---
    plt.title(label="Frequency spectrum (DFT)")
    plt.xlabel(xlabel="Frequency [Hz]")
    plt.ylabel(ylabel="Amplitude")
    plt.bar(x=usable_frequencies, height=amplitude, width=1)
    plt.show()


def main() -> None:
    run_dft_with_graphs()


if __name__ == "__main__":
    main()
