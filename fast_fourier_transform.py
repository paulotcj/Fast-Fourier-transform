import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import math
from typing import Any


#-------------------------------------------------------------------------
class Signal:
    """Generates and holds the composite time-domain signal."""
    #-------------------------------------------------------------------------
    def __init__(self, duration: float, num_samples: int) -> None:
        self.duration    : float = duration
        self.num_samples : int   = num_samples
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _compose_wave_values(self, time_span_samples: np.ndarray) -> np.ndarray:
        """Builds the composite signal as a sum of four phase-shifted cosine waves.
        The param time_span_samples is equidistant points in space/time where we want to sample the wave
        function.
        """
        return (
              np.cos(     2 * np.pi * time_span_samples - 1.571)   #  1 Hz
            + np.cos( 7 * 2 * np.pi * time_span_samples - 1.571)   #  7 Hz
            + np.cos(13 * 2 * np.pi * time_span_samples - 1.571)   # 13 Hz
            + np.cos(15 * 2 * np.pi * time_span_samples - 1.571)   # 15 Hz
        )
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def get_signal(self) -> dict[str, Any]:
        #-----
        time_span_samples_arr : np.ndarray = np.linspace(start=0, stop=self.duration, num=self.num_samples)  # evenly spaced numbers over a specified interval
        wave_values_arr       : np.ndarray = self._compose_wave_values(time_span_samples=time_span_samples_arr)
        step_interval         : float      = float(time_span_samples_arr[1] - time_span_samples_arr[0])
        sample_rate           : float      = 1.0 / step_interval
        #-----
        time_span_samples : list[float] = time_span_samples_arr.tolist()
        wave_values       : list[float] = wave_values_arr.tolist()
        #-----
        return_obj : dict[str, Any] = {
            'time_span_samples' : time_span_samples,
            'wave_values'       : wave_values,
            'num_samples'       : self.num_samples,
            'step_interval'     : step_interval,
            'sample_rate'       : sample_rate,
        }
        return return_obj
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
class FastFourierTransform:
    """
    Computes the Fast Fourier Transform using the Cooley-Tukey Radix-2
    Decimation-In-Time (DIT) algorithm.

    Compared to the O(N^2) DFT, this algorithm recursively splits the problem
    in half at each stage, reducing complexity to O(N log N).
    """
    #-------------------------------------------------------------------------
    def __init__(self, signal_data: dict[str, Any]) -> None:
        self._signal_wave_values : list[float] = signal_data['wave_values']
        self._signal_num_samples : int         = signal_data['num_samples']
        self._signal_sample_rate : float       = signal_data['sample_rate']
        # Radix-2 requires the input length to be a power of 2; compute it up front
        # so all subsequent methods share the same padded size.
        self._fft_size           : int         = self._next_power_of_two(n=self._signal_num_samples)

        self._result             : np.ndarray  = self._compute()
        self._magnitude          : np.ndarray  = self._compute_magnitude()
        self.usable_frequencies  : np.ndarray  = self._compute_usable_frequencies()
        self.amplitude           : np.ndarray  = self._compute_amplitude()
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _next_power_of_two(self, n: int) -> int:
        """
        Returns the smallest power of 2 that is >= n.
        The Cooley-Tukey Radix-2 algorithm requires the input length to be a
        power of 2; this is used to determine how much zero-padding is needed.

          e.g.  n=1024 -> 1024  (already a power of 2, no padding needed)
                n=1280 -> 2048
                n=5    -> 8
        """
        power : int = 1
        while power < n:
            power *= 2 # we could use bit shift: power << bitshift_val  , but this is more understandable
        return power
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _fft_recursive(self, values: list[complex]) -> list[complex]:
        """
        Cooley-Tukey Radix-2 DIT FFT — core recursive step.

        Splits the input into even- and odd-indexed samples, recursively
        computes the FFT of each half, then recombines using twiddle factors.

          X[k]       = E[k] + W_N^k * O[k]
          X[k + N/2] = E[k] - W_N^k * O[k]

          where  W_N^k = e^(-j*2*π*k/N)   (twiddle factor)
                 E[k]  = FFT of even-indexed samples: x[0], x[2], x[4], ...
                 O[k]  = FFT of odd-indexed  samples: x[1], x[3], x[5], ...
                 k     = 0, 1, ..., N/2 - 1

        The same twiddle factor W_N^k appears in both lines — the only difference
        is the sign. This is the "butterfly" operation that halves the work at
        every level of the recursion.
        """
        n : int = len(values)

        # Base case: the FFT of a single sample is the sample itself
        if n == 1:
            return list(values)

        # Split into even- and odd-indexed samples
        even_samples : list[complex] = values[0::2]   # x[0], x[2], x[4], ...
        odd_samples  : list[complex] = values[1::2]   # x[1], x[3], x[5], ...

        # Recursively compute FFT of each half
        even_fft : list[complex] = self._fft_recursive(values=even_samples)
        odd_fft  : list[complex] = self._fft_recursive(values=odd_samples)

        # Combine via butterfly: twiddle factor W_N^k = cos(-2πk/N) + j*sin(-2πk/N)
        result : list[complex] = [complex(real=0.0, imag=0.0)] * n
        half_n : int           = n // 2

        for k in range(half_n):
            twiddle_angle : float   = -2.0 * math.pi * k / n
            twiddle       : complex = complex(
                real = math.cos(twiddle_angle),
                imag = math.sin(twiddle_angle),
            )
            result[k]          = even_fft[k] + twiddle * odd_fft[k]
            result[k + half_n] = even_fft[k] - twiddle * odd_fft[k]

        return result
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _compute(self) -> np.ndarray:
        """
        Prepares the input and runs the FFT.

        Signal values (real numbers) are cast to complex with imaginary part = 0.
        If the signal length is not a power of 2, zeros are appended to reach
        the next power of 2. Zero-padding does not alter the signal; it
        interpolates the frequency axis, giving finer frequency resolution.

          e.g. signal length 1280 -> zero-padded to 2048 before the transform
        """
        complex_input : list[complex] = [complex(real=v, imag=0.0) for v in self._signal_wave_values]
        padding_size  : int           = self._fft_size - self._signal_num_samples
        complex_input                += [complex(real=0.0, imag=0.0)] * padding_size

        result_list : list[complex] = self._fft_recursive(values=complex_input)
        return np.array(result_list)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _compute_magnitude(self) -> np.ndarray:
        """
        Computes |Xk| = sqrt(a^2 + b^2) for each complex FFT coefficient.
        np.abs() is the equivalent numpy function.
        """
        magnitude : np.ndarray = np.array([0.0] * self._result.size)
        for i in range(self._result.size):
            magnitude[i] = math.sqrt(self._result[i].real**2 + self._result[i].imag**2)
        return magnitude
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _compute_usable_frequencies(self) -> np.ndarray:
        """
        Builds the full frequency axis from 0 Hz to the sample rate, then discards the upper half.

        The FFT of N real-valued samples produces N complex coefficients, but the upper half
        (indices N/2 to N-1) are mirror images of the lower half and carry no new information.
        The highest frequency that can be faithfully represented is sample_rate / 2, known as
        the Nyquist frequency. Everything above it is discarded.

          Example — N=8 samples, sample_rate=8 Hz:
            Full frequency axis:   [0, 1, 2, 3, 4, 5, 6, 7]  (8 bins, spacing = 1 Hz)
            After Nyquist cutoff:  [0, 1, 2, 3]               (keep only the first 4 = N//2)
            Bins 4-7 are the complex conjugate mirror of bins 0-3 and are dropped.
        """
        frequencies : np.ndarray = np.linspace(
            start = 0,
            stop  = self._signal_sample_rate,
            num   = self._fft_size,
        )
        return frequencies[0 : self._fft_size // 2]
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _compute_amplitude(self) -> np.ndarray:
        """
        Normalizes magnitude over the usable half of the spectrum.
        Multiplies by 2 to compensate for discarding the upper half (Nyquist cutoff),
        then divides by the original signal length N to scale back to the original
        signal amplitude. The original length (not the zero-padded size) is used
        because the padding contributes no real signal energy.

          e.g. a 1 Hz tone with N=8 yields raw magnitude ~3.9997
               3.9997 * 2 / 8 = 0.9999 ≈ 1.0
        """
        return self._magnitude[0 : self._fft_size // 2] * 2 / self._signal_num_samples
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def get_transform(self) -> dict[str, Any]:
        return_obj : dict[str, Any] = {
            'usable_frequencies' : self.usable_frequencies.tolist(),
            'amplitude'          : self.amplitude.tolist(),
        }
        return return_obj
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
class Plotter:
    """Renders the time-domain and frequency-domain graphs."""
    #-------------------------------------------------------------------------
    def __init__(self, signal_data: dict[str, Any], fft_data: dict[str, Any]) -> None:
        self._signal_time_span_samples : list[float] = signal_data['time_span_samples']
        self._signal_values            : list[float] = signal_data['wave_values']
        self._usable_frequencies       : list[float] = fft_data['usable_frequencies']
        self._amplitude                : list[float] = fft_data['amplitude']
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _plot_time_domain(self, ax: Axes) -> None:
        """Plots the composite signal amplitude over time."""
        ax.set_title(label="Time-domain signal")
        ax.set_xlabel(xlabel="Time [s]")
        ax.set_ylabel(ylabel="Amplitude")
        ax.plot(self._signal_time_span_samples, self._signal_values)
        ax.grid(True)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _plot_frequency_spectrum(self, ax: Axes) -> None:
        """Plots the FFT magnitude spectrum up to the Nyquist frequency, with peak annotations."""
        bar_width : float = self._usable_frequencies[1] - self._usable_frequencies[0]

        ax.set_title(label="Frequency spectrum (Fast Fourier Transform)")
        ax.set_xlabel(xlabel="Frequency [Hz]")
        ax.set_ylabel(ylabel="Amplitude")
        ax.bar(x=self._usable_frequencies, height=self._amplitude, width=bar_width)
        ax.grid(True, axis='y')

        # Annotate local maxima that exceed 10% of the peak amplitude,
        # and track the highest peak frequency to set a tight x-axis limit.
        peak_threshold    : float = max(self._amplitude) * 0.1
        highest_peak_freq : float = 0.0
        for i in range(1, len(self._amplitude) - 1):
            if (    self._amplitude[i] > self._amplitude[i - 1]
                and self._amplitude[i] > self._amplitude[i + 1]
                and self._amplitude[i] > peak_threshold         ):
                ax.annotate(
                    text       = f'{self._usable_frequencies[i]:.0f} Hz',
                    xy         = (self._usable_frequencies[i], self._amplitude[i]),
                    xytext     = (0, 6),
                    textcoords = 'offset points',
                    ha         = 'center',
                    fontsize   = 9,
                )
                if self._usable_frequencies[i] > highest_peak_freq:
                    highest_peak_freq = self._usable_frequencies[i]

        # Limit x-axis to 20% beyond the highest detected peak so the plot
        # doesn't show a large empty frequency range.
        ax.set_xlim(left=0, right=highest_peak_freq * 1.2)
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def plot_all(self) -> None:
        """Renders both plots stacked in a single figure."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self._plot_time_domain(ax=ax1)
        self._plot_frequency_spectrum(ax=ax2)
        plt.tight_layout()
        plt.show()
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def main() -> None:
    #-----
    signal_duration    : float = 0.875 * 5  # initial duration set to 0.875 but decided to lengthen by multiplying by 5
    signal_num_samples : int   = 8 * 5 * 32 # originally 8 samples for a duration of 0.875, once the signal was lengthened by 5 we then multiplied by 5 and then later we multiplied by 32 to add more data
    #-----
    signal      : Signal             = Signal(duration=signal_duration, num_samples=signal_num_samples)
    signal_data : dict[str, Any]     = signal.get_signal()
    #-----
    fft         : FastFourierTransform = FastFourierTransform(signal_data=signal_data)
    fft_data    : dict[str, Any]       = fft.get_transform()
    #-----
    plotter     : Plotter            = Plotter(signal_data=signal_data, fft_data=fft_data)
    plotter.plot_all()
#-------------------------------------------------------------------------

if __name__ == "__main__":
    main()
