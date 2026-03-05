# The Fast Fourier Transform — A Guided Explanation

This document explains the Fast Fourier Transform (FFT) from the ground up, using
`fast_fourier_transform.py` as a concrete reference throughout. Every concept is
tied directly to a piece of code or a formula you can find in that file.

---

## Table of Contents

1. [What problem does the FFT solve?](#1-what-problem-does-the-fft-solve)
2. [The signal we are analysing](#2-the-signal-we-are-analysing)
3. [From continuous to discrete — the DFT](#3-from-continuous-to-discrete--the-dft)
4. [Why DFT is slow — O(N²)](#4-why-dft-is-slow--on²)
5. [The FFT insight — divide and conquer](#5-the-fft-insight--divide-and-conquer)
6. [The twiddle factor](#6-the-twiddle-factor)
7. [The butterfly operation](#7-the-butterfly-operation)
8. [A worked example — N=8](#8-a-worked-example--n8)
9. [Zero-padding and the power-of-2 requirement](#9-zero-padding-and-the-power-of-2-requirement)
10. [From complex output to magnitude](#10-from-complex-output-to-magnitude)
11. [The Nyquist cutoff and usable frequencies](#11-the-nyquist-cutoff-and-usable-frequencies)
12. [Amplitude normalisation](#12-amplitude-normalisation)
13. [Reading the output](#13-reading-the-output)
14. [The full pipeline in code](#14-the-full-pipeline-in-code)
15. [Recursive vs iterative — a note on production use](#15-recursive-vs-iterative--a-note-on-production-use)

---

## 1. What problem does the FFT solve?

Any real-world signal — sound, vibration, radio waves, electrical current — is
recorded as a sequence of measurements over time. Looking at those measurements
directly (the **time domain**) tells you *when* something happened, but not *what
frequencies* are present.

The Fourier Transform answers: **"which frequencies make up this signal, and how
strong is each one?"**

Consider a musical chord. In the time domain it looks like a complicated squiggle.
In the **frequency domain** it resolves into a small set of peaks — one per note —
each with a height equal to that note's volume. The FFT is the algorithm that
performs this decomposition efficiently.

---

## 2. The signal we are analysing

The `Signal` class in the code constructs a synthetic composite signal — a sum of
four cosine waves at known frequencies, each with amplitude 1.0:

```python
np.cos(     2 * np.pi * time_span_samples - 1.571)   #  1 Hz
np.cos( 7 * 2 * np.pi * time_span_samples - 1.571)   #  7 Hz
np.cos(13 * 2 * np.pi * time_span_samples - 1.571)   # 13 Hz
np.cos(15 * 2 * np.pi * time_span_samples - 1.571)   # 15 Hz
```

Each term follows the general cosine form:

```
A · cos(2·π·f·t - φ)
```

where:
- `A` = amplitude (1.0 for all four waves here)
- `f` = frequency in Hz (cycles per second): 1, 7, 13, 15
- `t` = time in seconds
- `φ` = phase offset (1.571 ≈ π/2, shifting each cosine to start as a sine shape)

When you add all four together, the result looks like an irregular, noisy wave.
The job of the FFT is to look at that composite wave and recover the four original
frequencies and their amplitudes — without knowing them in advance.

The signal is sampled over **4.375 seconds** with **1280 samples**:

```python
signal_duration    = 0.875 * 5   # = 4.375 seconds
signal_num_samples = 8 * 5 * 32  # = 1280 samples
```

The time between consecutive samples (`step_interval`) and the number of samples
per second (`sample_rate`) are derived in `Signal.get_signal()`:

```python
step_interval = time_span_samples_arr[1] - time_span_samples_arr[0]
sample_rate   = 1.0 / step_interval
```

---

## 3. From continuous to discrete — the DFT

The **continuous Fourier Transform** operates on a function of time and produces a
function of frequency:

```
X(F) = ∫[-∞, ∞] x(t) · e^(-j·2·π·F·t) · dt
```

Because a computer can only work with finite, discrete samples, we use the
**Discrete Fourier Transform (DFT)** instead:

```
X[k] = SUM[n=0..N-1]  x[n] · e^(-j·2·π·k·n / N)
```

Symbol definitions:
| Symbol | Meaning |
|--------|---------|
| `x[n]` | the n-th sample of the input signal |
| `n`    | sample index in the time domain, 0 to N-1 |
| `N`    | total number of samples |
| `k`    | frequency bin index in the output, 0 to N-1 |
| `X[k]` | the complex number describing the strength and phase at frequency bin k |
| `j`    | the imaginary unit (j² = -1); written as `j` not `i` to avoid confusion with indices |

The DFT computes one output value `X[k]` for each frequency bin `k`. Each output
is a complex number because a sinusoidal component has both an amplitude and a
phase.

---

## 4. Why DFT is slow — O(N²)

To compute all N output values directly from the DFT formula, you need:
- An outer loop over k: **N iterations**
- An inner loop over n for each k: **N iterations**

Total: **N × N = N² multiplications and additions**.

For our signal with N=1280 that is ~1.6 million operations. For N=1,000,000 (common
in audio or signal processing) that is **one trillion** operations — far too slow
for real-time use.

The `DiscreteFourierTransform` class in `discrete_fourier_transform.py` implements
this directly as a double nested loop and serves as the baseline for comparison.

---

## 5. The FFT insight — divide and conquer

The key observation behind the FFT is that the DFT sum can be **split into two
independent half-size DFTs**:

```
X[k] = SUM[n=0..N-1]  x[n] · e^(-j·2·π·k·n / N)
     = SUM[n even]     x[n] · e^(-j·2·π·k·n / N)    ← even-indexed samples
     + SUM[n odd]      x[n] · e^(-j·2·π·k·n / N)    ← odd-indexed samples
```

To simplify, introduce index `m` that counts only within each half
(m = 0, 1, ..., N/2 - 1):

- **Even terms**: every even `n` is a multiple of 2, so write `n = 2m`
- **Odd terms**: every odd `n` is one past a multiple of 2, so write `n = 2m + 1`

After substituting and simplifying, the exponent `e^(-j·2·π·k·(2m)/N)` factors
because `2/N` reduces to `1/(N/2)`, which is the twiddle factor of a half-size DFT.
This gives:

```
X[k] = E[k]  +  e^(-j·2·π·k/N)  ·  O[k]
```

where:
- `E[k]` = DFT of the even-indexed samples: x[0], x[2], x[4], ...
- `O[k]` = DFT of the odd-indexed samples:  x[1], x[3], x[5], ...

In `_fft_recursive()` this split is:

```python
even_samples_complex = values_complex_input[0::2]   # x[0], x[2], x[4], ...
odd_samples_complex  = values_complex_input[1::2]   # x[1], x[3], x[5], ...

even_fft = self._fft_recursive(values_complex_input=even_samples_complex)
odd_fft  = self._fft_recursive(values_complex_input=odd_samples_complex)
```

Each of those two recursive calls does the same split again on its own input. This
continues until the sub-problem has only 1 sample — the base case, where the FFT of
a single sample is the sample itself:

```python
if len_values_complex_input == 1:
    return list(values_complex_input)
```

The recursion tree for N=8 looks like this:

```
                     [x0,x1,x2,x3,x4,x5,x6,x7]   N=8
                           /              \
             [x0,x2,x4,x6]              [x1,x3,x5,x7]   N=4
               /       \                  /       \
         [x0,x4]     [x2,x6]        [x1,x5]     [x3,x7]   N=2
          /   \       /   \          /   \        /   \
        [x0] [x4] [x2] [x6]       [x1] [x5]  [x3] [x7]   N=1 (base case)
```

The total number of levels is **log₂(N)** — for N=8 that is 3 levels.

---

## 6. The twiddle factor

The term `e^(-j·2·π·k/N)` is called the **twiddle factor**, written `W_N^k`:

```
W_N^k = e^(-j·2·π·k/N)
```

Using Euler's formula (`e^(j·x) = cos(x) + j·sin(x)`), this expands to:

```
W_N^k = cos(-2·π·k/N)  +  j·sin(-2·π·k/N)
```

Geometrically, this is a point on the **unit circle** in the complex plane. As `k`
increases from 0 to N/2, the point rotates clockwise around the circle.

- At k=0: W = cos(0) + j·sin(0) = 1 + 0j  (no rotation)
- At k=1: W rotates by one step of 2π/N clockwise
- At k=N/4: W has rotated 90° — W = 0 - j

It acts as a **phase shift**: it rotates the odd half's frequency components so they
align correctly with the even half's before the two are added together.

In code:

```python
twiddle_angle = -2.0 * math.pi * k / len_values_complex_input
twiddle       = complex(
    real = math.cos(twiddle_angle),   # real part of W_N^k
    imag = math.sin(twiddle_angle),   # imaginary part of W_N^k
)
```

---

## 7. The butterfly operation

Once `E[k]` and `O[k]` are known, the full output is assembled using what is called
a **butterfly operation**. It produces two output bins from one twiddle multiplication:

```
X[k]       = E[k]  +  W_N^k · O[k]     ← upper output
X[k + N/2] = E[k]  -  W_N^k · O[k]     ← lower output
```

The lower output is free — it reuses the same twiddle product with only a sign flip.
This works because of a symmetry property of the twiddle factor:

```
W_N^(k + N/2) = -W_N^k
```

Proof: substituting k + N/2 into the definition:
```
e^(-j·2·π·(k + N/2)/N) = e^(-j·2·π·k/N) · e^(-j·π) = W_N^k · (-1) = -W_N^k
```

Since `e^(-j·π) = -1` (half a rotation around the unit circle), the twiddle factor
at index `k + N/2` is exactly the negative of the one at `k`.

Diagram:

```
E[k] ────┬──────────────────► X[k]       = E[k] + W · O[k]
         │        ↗ (+)
         │   W · O[k]
         │        ↘ (-)
O[k] ────┴──────────────────► X[k+N/2]  = E[k] - W · O[k]
```

In code:

```python
result_complex[k]          = even_fft[k] + twiddle * odd_fft[k]   # upper output
result_complex[k + half_n] = even_fft[k] - twiddle * odd_fft[k]   # lower output
```

**Why this is O(N log N):** at each level of the recursion there are N/2 butterflies,
each costing one complex multiplication. There are log₂(N) levels. Total cost:
`(N/2) · log₂(N)` multiplications — far less than the N² of the direct DFT.

| N      | DFT operations (N²) | FFT operations (N/2 · log₂N) |
|--------|---------------------|-------------------------------|
| 8      | 64                  | 12                            |
| 1,024  | 1,048,576           | 5,120                         |
| 1,280  | 1,638,400           | ~5,500 (after padding to 2048)|
| 1,000,000 | 10¹²           | ~10,000,000                   |

---

## 8. A worked example — N=8

Suppose the input is 8 samples: `[x0, x1, x2, x3, x4, x5, x6, x7]`.

**Step 1 — split:**
```
Even: [x0, x2, x4, x6]
Odd:  [x1, x3, x5, x7]
```

**Step 2 — split each again:**
```
Even of even: [x0, x4]    Even of odd: [x1, x5]
Odd of even:  [x2, x6]    Odd of odd:  [x3, x7]
```

**Step 3 — split to size 1 (base case):**
```
[x0], [x4], [x2], [x6], [x1], [x5], [x3], [x7]
```
Each of these is its own FFT (a single complex number).

**Step 4 — combine upward (butterflies):**

Level 1 — N=2 butterflies (twiddle W_2^k, k=0 only, W_2^0 = 1):
```
FFT([x0,x4]): X[0] = x0 + x4,   X[1] = x0 - x4
FFT([x2,x6]): X[0] = x2 + x6,   X[1] = x2 - x6
FFT([x1,x5]): X[0] = x1 + x5,   X[1] = x1 - x5
FFT([x3,x7]): X[0] = x3 + x7,   X[1] = x3 - x7
```

Level 2 — N=4 butterflies (twiddle W_4^k, k=0,1):
```
FFT([x0,x2,x4,x6]) uses FFT([x0,x4]) and FFT([x2,x6])
FFT([x1,x3,x5,x7]) uses FFT([x1,x5]) and FFT([x3,x7])
```

Level 3 — N=8 butterfly (twiddle W_8^k, k=0..3):
```
FFT([x0..x7]) uses FFT([x0,x2,x4,x6]) and FFT([x1,x3,x5,x7])
→ Final 8 output bins X[0] through X[7]
```

Total butterfly operations: 4 + 4 + 4 = 12, compared to 8² = 64 for the direct DFT.

---

## 9. Zero-padding and the power-of-2 requirement

The Cooley-Tukey Radix-2 algorithm requires N to be a **power of 2**, because every
split must produce two equal halves.

Our signal has 1280 samples. Since 1280 is not a power of 2, `_next_power_of_two()`
finds the next one:

```python
def _next_power_of_two(self, n: int) -> int:
    power = 1
    while power < n:
        power *= 2
    return power
# 1280 → 2048
```

The gap is filled with zeros (**zero-padding**) in `_compute()`:

```python
complex_input  = [complex(real=v, imag=0.0) for v in self._signal_wave_values]
padding_size   = self._fft_size - self._signal_num_samples
complex_input += [complex(real=0.0, imag=0.0)] * padding_size
```

Zero-padding does **not** add information. It interpolates the frequency axis,
producing more frequency bins and therefore a smoother-looking spectrum — but the
underlying frequency resolution is determined by the original signal length and
sample rate.

Note also that the signal values (originally real numbers) are cast to complex with
`imag=0.0`. The FFT algorithm operates on complex inputs, and a real signal is just
a special case where the imaginary part happens to be zero.

---

## 10. From complex output to magnitude

The FFT produces an array of **complex numbers** — one per frequency bin. A complex
number `a + j·b` has two components:

- `a` (real part) — related to the cosine component at that frequency
- `b` (imaginary part) — related to the sine component at that frequency

Neither component alone tells you how strong the frequency is, because the energy is
distributed across both. The **magnitude** combines them into a single non-negative
number using the Pythagorean theorem:

```
|X[k]| = sqrt(a² + b²)
```

This is the length of the vector `(a, b)` in the complex plane — how far the point
is from the origin, regardless of its angle (phase).

```
         imaginary
             │
           b ┼ · · · · ●  (a + j·b)
             │        /│
             │  |X[k]/ │
             │      /  │ b
             │    /    │
             │  /      │
─────────────┼─────────┼──────── real
             0    a
             │←── a ──►│
```

In code (`_compute_magnitude()`):

```python
magnitude[i] = math.sqrt(self._result[i].real**2 + self._result[i].imag**2)
```

---

## 11. The Nyquist cutoff and usable frequencies

The FFT of N real-valued samples produces N complex coefficients, but **the upper
half (bins N/2 to N-1) are exact mirror images of the lower half**. They carry no
new information and are discarded.

The reason: for a real-valued signal, the DFT has a symmetry property called
**conjugate symmetry** — `X[N-k] = X[k]*` (complex conjugate). This means bin N-1
mirrors bin 1, bin N-2 mirrors bin 2, and so on.

The highest frequency that can be faithfully represented is `sample_rate / 2`,
known as the **Nyquist frequency**. Above that, the sampling is too coarse to
distinguish the frequency from a lower one — an effect called **aliasing**.

Example with N=8, sample_rate=8 Hz:

```
Full output bins:  [X0,  X1,  X2,  X3,  X4,  X5,  X6,  X7]
Frequencies (Hz):  [ 0,   1,   2,   3,   4,   5,   6,   7]
                    └──── usable ────┘└──── mirror (discard) ────┘
                          bins 0-3              bins 4-7
Nyquist = 8/2 = 4 Hz
```

In `_compute_usable_frequencies()`:

```python
frequencies = np.linspace(start=0, stop=self._signal_sample_rate, num=self._fft_size)
return frequencies[0 : self._fft_size // 2]
```

---

## 12. Amplitude normalisation

After discarding the upper half, the raw magnitudes still need to be scaled to match
the amplitude of the original signal waves.

**What is amplitude?**

For a cosine wave written as `A · cos(2·π·f·t)`, `A` is the amplitude — how tall
the wave is, i.e. how far it swings above and below zero. Our signal has four
cosine components each with `A = 1.0` at 1, 7, 13, and 15 Hz.

After the FFT and normalisation, the amplitude at each bin should read ≈ 1.0 at
those four frequencies and ≈ 0.0 everywhere else — confirming both which frequencies
are present and how strong each one is.

Two corrections are needed:

**1 — Divide by N**

The DFT formula sums N sample contributions into each coefficient, so raw magnitudes
are proportional to N. Dividing by N scales them back.

The original signal length is used — not the zero-padded size — because the padding
contributes zeros that add no real energy. Using the padded size would
artificially shrink the result.

**2 — Multiply by 2**

Discarding the upper half of the spectrum also discards half the energy (since the
two halves are symmetric mirrors). Multiplying by 2 restores what was lost.

Combined correction applied in `_compute_amplitude()`:

```python
return self._magnitude[0 : self._fft_size // 2] * 2 / self._signal_num_samples
```

Concrete example:
```
Pure 1 Hz cosine, amplitude = 1.0, N = 8 samples
  Raw FFT magnitude at bin 1  ≈ 3.9997
  × 2                          = 7.9994   (restore discarded mirror)
  ÷ 8                          = 0.9999   ≈ 1.0  ✓
```

---

## 13. Reading the output

After running the full pipeline, `get_transform()` returns two lists:

```python
{
    'usable_frequencies': [...],   # frequency in Hz for each bin
    'amplitude':          [...],   # signal amplitude at each frequency
}
```

For our composite signal (1 + 7 + 13 + 15 Hz, all amplitude 1.0), the output looks
roughly like this:

```
Amplitude
  1.0 │    █              █         █    █
      │    │              │         │    │
  0.5 │    │              │         │    │
      │    │              │         │    │
  0.0 ┼────┼──────────────┼─────────┼────┼────► Frequency (Hz)
       0   1   3   5   7   9  11  13  15  18
```

Four peaks of height ≈ 1.0 at exactly 1, 7, 13, and 15 Hz. Silence (≈ 0.0)
everywhere else. The FFT has successfully decomposed the composite signal back
into its four individual components.

The `Plotter` class detects these peaks automatically by finding local maxima above
10% of the highest peak, annotates them with their frequency, and clips the x-axis
20% beyond the highest peak so no empty space is shown.

---

## 14. The full pipeline in code

`main()` wires the three classes together. Each stage produces plain data (a dict)
that is passed to the next stage — no class depends directly on another:

```python
# 1. Generate the signal
signal      = Signal(duration=0.875 * 5, num_samples=8 * 5 * 32)
signal_data = signal.get_signal()
#   signal_data keys: time_span_samples, wave_values,
#                     num_samples, step_interval, sample_rate

# 2. Run the FFT
fft      = FastFourierTransform(signal_data=signal_data)
fft_data = fft.get_transform()
#   fft_data keys: usable_frequencies, amplitude

# 3. Plot
plotter = Plotter(signal_data=signal_data, fft_data=fft_data)
plotter.plot_all()
```

Inside `FastFourierTransform.__init__()`, the steps run in order:

```
signal_data
     │
     ▼
_compute()               → raw complex FFT output  (_result)
     │
     ▼
_compute_magnitude()     → |X[k]| for each bin     (_magnitude)
     │
     ▼
_compute_usable_frequencies() → frequency axis, Nyquist-trimmed
     │
_compute_amplitude()     → normalised amplitude (× 2 / N)
     │
     ▼
get_transform()          → { usable_frequencies, amplitude }
```

---

## 15. Recursive vs iterative — a note on production use

The implementation here uses a recursive Cooley-Tukey approach. This maps closely
to the mathematical definition and is easy to follow, which makes it a good fit for
learning.

However, recursive implementations are generally avoided in production systems:

- Function call overhead accumulates across log₂(N) recursion levels
- Iterative implementations have better cache behaviour and are easier to vectorize
  with SIMD instructions (SSE, AVX, NEON)

The standard production alternative is an iterative Cooley-Tukey using:

1. **Bit-reversal permutation** — reorders the input into the sequence it would
   naturally reach at the leaves of the recursion tree. Index bits are reversed, e.g.
   for N=8: index 3 (binary `011`) maps to index 6 (binary `110`).

2. **Bottom-up butterfly passes** — log₂(N) passes over the array, each doubling
   the stride and combining element pairs with twiddle factors, exactly as the
   recursion does but without any function calls.

Battle-tested libraries go further:
- **FFTW** and **Intel MKL** — mixed-radix algorithms (any N, no zero-padding needed),
  SIMD vectorization, multi-threading, and auto-tuning at runtime
- **cuFFT** — GPU-accelerated FFT
- **numpy's pocketfft** (used by `np.fft.fft`) — optimised iterative mixed-radix

For this project, the recursive approach is preferred for clarity. If you want to
compare results, `np.fft.fft(values)` is the one-line equivalent of the entire
`FastFourierTransform` class.
