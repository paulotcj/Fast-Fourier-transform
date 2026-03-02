# Discrete Fourier Transform

A from-scratch Python implementation of the Discrete Fourier Transform (DFT) for learning purposes.

## What it does

Generates a composite signal made of four cosine waves (1, 7, 13, and 15 Hz), applies the DFT manually, and plots:
1. The time-domain signal
2. The frequency-domain magnitude spectrum, showing the four component frequencies as peaks

## Running

```bash
python dft.py
```

**Dependencies:** `numpy`, `matplotlib`

## Implementation notes

- `discrete_fourier_transform()` is an O(N²) implementation using Euler's formula. `np.fft.fft()` is left as a commented reference for comparison.
- `complex_absolute_value()` computes `|z| = sqrt(a² + b²)` manually. `np.abs()` is left as a commented reference.
- Amplitude is normalized by multiplying by `2/N`: the factor of 2 compensates for discarding the upper half of the spectrum (Nyquist cutoff), and `1/N` scales the result back to the original signal amplitude.
