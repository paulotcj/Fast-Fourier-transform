# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A Python learning project implementing the Discrete Fourier Transform (DFT) from scratch. The goal is educational — understanding the math behind DFT rather than using optimized libraries.

## Running

```bash
python dft.py
```

Dependencies: `numpy`, `matplotlib`. Running the script opens two matplotlib windows sequentially: a time-domain signal plot and a frequency-domain magnitude spectrum.

## Architecture

Everything lives in `dft.py`:

- `complex_absolute_value(values)` — manual implementation of `|z| = sqrt(a² + b²)` for complex arrays (equivalent to `np.abs`)
- `discrete_fourier_transform(values)` — O(N²) DFT using Euler's formula; `np.fft.fft()` is left commented as an O(N log N) reference
- `run_dft_with_graphs()` — builds a composite cosine signal (1, 7, 13, 15 Hz), runs the DFT, applies Nyquist cutoff, normalizes magnitude, and renders both graphs
- `main()` — entry point

`fft.xlsx` is a companion spreadsheet used for manual calculations and cross-checking results.

## Key formulas

- DFT: `Xk = SUM[n=0..N-1] xn * e^((-j*2*π*k*n)/N)`
- Euler's formula expansion: `e^(jx) = cos(x) + j*sin(x)`
- Amplitude normalization: `|Xk[0..N/2]| * 2 / N` (factor of 2 compensates for Nyquist cutoff, 1/N scales to original amplitude)
