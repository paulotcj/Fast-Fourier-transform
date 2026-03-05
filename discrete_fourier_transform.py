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
    def _compose_wave_values(self,time_span_samples: np.ndarray) -> np.ndarray:
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
    def get_signal(self) -> dict[str,Any]:
        #-----
        time_span_samples_arr : np.ndarray = np.linspace(start=0, stop=self.duration, num=self.num_samples) # evenly spaced numbers over a specified interval
        wave_values_arr       : np.ndarray = self._compose_wave_values(time_span_samples=time_span_samples_arr)
        step_interval         : float      = float(time_span_samples_arr[1] - time_span_samples_arr[0])
        sample_rate           : float      = 1.0 / step_interval
        #-----
        time_span_samples : list[float] = time_span_samples_arr.tolist()
        wave_values       : list[float] = wave_values_arr.tolist()
        #-----

        return_obj : dict[str,Any] = {
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
class DiscreteFourierTransform:
    """Computes the Discrete Fourier Transform of a Signal and exposes the results."""
    #-------------------------------------------------------------------------
    def __init__(self, signal_data  : dict[str, Any]) -> None:

        self._signal_wave_values    : list[float]   = signal_data['wave_values']
        self._signal_num_samples    : int           = signal_data['num_samples']
        self._signal_sample_rate    : float         = signal_data['sample_rate']

        self._result                : np.ndarray    = self._compute()
        self._magnitude             : np.ndarray    = self._compute_magnitude()
        self.usable_frequencies     : np.ndarray    = self._compute_usable_frequencies()
        self.amplitude              : np.ndarray    = self._compute_amplitude()
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _compute(self) -> np.ndarray:
        """
        O(N^2) DFT implementation using Euler's formula.
        np.fft.fft() provides an optimized O(N log N) equivalent.

          Continuous Fourier Transform:  X(F) = ∫[-∞,∞] x(t) * e^(-j*2*π*F*t) * dt
          Discrete Fourier Transform:    Xk   = SUM[n=0..N-1] xn * e^((-j*2*π*k*n)/N)
             k/N ~ F  (frequency index maps to frequency)
             n   ~ t  (sample index maps to time)
        """
        num_samples         : int           = self._signal_num_samples
        neg_two_pi_over_n   : float         = -math.pi * 2 / num_samples
        result_real_imag    : np.ndarray    = np.array([0 + 0j] * num_samples) #initializes an array or complex numbers (all zeroes)

        #-----
        for k in range(num_samples):
            sum_real: float = 0.0
            sum_imag: float = 0.0
            #-----
            for n in range(num_samples):
                exp_value: float = neg_two_pi_over_n * k * n
                # Euler's formula: e^(j*x) = cos(x) + j*sin(x)
                sum_real += self._signal_wave_values[n] * math.cos(exp_value)
                sum_imag += self._signal_wave_values[n] * math.sin(exp_value)
            #-----
            result_real_imag[k] = complex(real=sum_real, imag=sum_imag)
        #-----

        return result_real_imag
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _compute_magnitude(self) -> np.ndarray:
        """
        Computes |Xk| = sqrt(a^2 + b^2) for each complex DFT coefficient.
        np.abs() is the equivalent numpy function.
        """
        magnitude: np.ndarray = np.array([0.0] * self._result.size)
        for i in range(self._result.size):
            magnitude[i] = math.sqrt(self._result[i].real**2 + self._result[i].imag**2)
        return magnitude
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _compute_usable_frequencies(self) -> np.ndarray:
        """
        Builds the full frequency axis from 0 Hz to the sample rate, then discards the upper half.

        The DFT of N real-valued samples produces N complex coefficients, but the upper half
        (indices N/2 to N-1) are mirror images of the lower half and carry no new information.
        The highest frequency that can be faithfully represented is sample_rate / 2, known as
        the Nyquist frequency. Everything above it is discarded.

          Example — N=8 samples, sample_rate=8 Hz:
            Full frequency axis:   [0, 1, 2, 3, 4, 5, 6, 7]  (8 bins, spacing = 1 Hz)
            After Nyquist cutoff:  [0, 1, 2, 3]               (keep only the first 4 = N//2)
            Bins 4-7 are the complex conjugate mirror of bins 0-3 and are dropped.
        """
        frequencies: np.ndarray = np.linspace(
            start = 0,
            stop  = self._signal_sample_rate,
            num   = self._signal_num_samples,
        )
        return frequencies[0 : self._signal_num_samples // 2]
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def _compute_amplitude(self) -> np.ndarray:
        """
        Normalizes magnitude over the usable half of the spectrum.
        Multiplies by 2 to compensate for discarding the upper half (Nyquist cutoff),
        then divides by N to scale back to the original signal amplitude.

          e.g. a 1 Hz tone with N=8 yields raw magnitude ~3.9997
               3.9997 * 2 / 8 = 0.9999 ≈ 1.0
        """
        return self._magnitude[0 : self._signal_num_samples // 2] * 2 / self._signal_num_samples
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
    def __init__(self, signal_data: dict[str, Any], dft_data: dict[str, Any]) -> None:
        self._signal_time_span_samples  : list[float]   = signal_data['time_span_samples']
        self._signal_values             : list[float]   = signal_data['wave_values']
        self._usable_frequencies        : list[float]   = dft_data['usable_frequencies']
        self._amplitude                 : list[float]   = dft_data['amplitude']
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
        """Plots the DFT magnitude spectrum up to the Nyquist frequency, with peak annotations."""
        bar_width : float = self._usable_frequencies[1] - self._usable_frequencies[0]

        ax.set_title(label="Frequency spectrum (Discrete Fourier Transform)")
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
    signal_duration     : float = 0.875 * 5 # initial duration set to 0.875 but decided to lengthen by multiplying by 5
    signal_num_samples  : int   = 8 * 5 * 32 # originall 8 sambles for a duration of 0.875, once the signal was lengthen by 5 wr then multiplied by 5 and then later we multiplied by 32 to add more data
    #-----
    signal: Signal                  = Signal(duration=signal_duration, num_samples=signal_num_samples)
    signal_data: dict[str, Any]     = signal.get_signal()
    #-----
    dft: DiscreteFourierTransform   = DiscreteFourierTransform(signal_data=signal_data)
    dft_data: dict[str, Any]        = dft.get_transform()
    #-----
    plotter: Plotter                = Plotter(signal_data=signal_data, dft_data=dft_data)
    plotter.plot_all()
#-------------------------------------------------------------------------

if __name__ == "__main__":
    main()
