import numpy as np
import matplotlib.pyplot as plt
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
    def _compose_values(self,time_span: np.ndarray) -> np.ndarray:
        """Builds the composite signal as a sum of four phase-shifted cosine waves."""
        return (
              np.cos(     2 * np.pi * time_span - 1.571)   #  1 Hz
            + np.cos( 7 * 2 * np.pi * time_span - 1.571)   #  7 Hz
            + np.cos(13 * 2 * np.pi * time_span - 1.571)   # 13 Hz
            + np.cos(15 * 2 * np.pi * time_span - 1.571)   # 15 Hz
        )
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def get_signal(self) -> dict[str,Any]:
        #-----
        time_span     : np.ndarray  = np.linspace(start=0, stop=self.duration, num=self.num_samples)
        values        : np.ndarray  = self._compose_values(time_span=self.time_span)
        num_samples   : int         = self.values.size
        step_interval : float       = float(self.time_span[1] - self.time_span[0])  # seconds per sample
        sample_rate   : float       = 1.0 / self.step_interval                      # samples per second (Hz)
        #-----
        print('hi')

        return_obj : dict[str,Any] = {
            'time_span'         : time_span,
            'values'            : values,
            'num_samples'       : num_samples,
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
    def __init__(self, signal: Signal) -> None:
        self._signal: Signal                = signal
        self._result: np.ndarray            = self._compute()
        self._magnitude: np.ndarray         = self._compute_magnitude()
        self.usable_frequencies: np.ndarray = self._compute_usable_frequencies()
        self.amplitude: np.ndarray          = self._compute_amplitude()
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
        num_samples: int         = self._signal.num_samples
        neg_two_pi_over_n: float = -math.pi * 2 / num_samples
        result: np.ndarray       = np.array([0 + 0j] * num_samples)

        for k in range(num_samples):
            sum_real: float = 0.0
            sum_imag: float = 0.0
            for n in range(num_samples):
                exp_value: float = neg_two_pi_over_n * k * n
                # Euler's formula: e^(j*x) = cos(x) + j*sin(x)
                sum_real += self._signal.values[n] * math.cos(exp_value)
                sum_imag += self._signal.values[n] * math.sin(exp_value)
            result[k] = complex(real=sum_real, imag=sum_imag)

        return result
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
        """Returns the frequency bins up to the Nyquist cutoff (lower half of the spectrum)."""
        frequencies: np.ndarray = np.linspace(
            start=0,
            stop=self._signal.sample_rate,
            num=self._signal.num_samples,
        )
        return frequencies[0 : self._signal.num_samples // 2]
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
        return self._magnitude[0 : self._signal.num_samples // 2] * 2 / self._signal.num_samples
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
class Plotter:
    """Renders the time-domain and frequency-domain graphs."""
    #-------------------------------------------------------------------------
    def __init__(self, signal: Signal, dft: DiscreteFourierTransform) -> None:
        self._signal: Signal = signal
        self._dft: DiscreteFourierTransform       = dft
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def plot_time_domain(self) -> None:
        """Plots the composite signal amplitude over time."""
        plt.title(label="Time-domain signal")
        plt.xlabel(xlabel="Time [s]")
        plt.ylabel(ylabel="Amplitude")
        plt.plot(self._signal.time_span, self._signal.values)
        plt.show()
    #-------------------------------------------------------------------------
    #-------------------------------------------------------------------------
    def plot_frequency_spectrum(self) -> None:
        """Plots the DFT magnitude spectrum up to the Nyquist frequency."""
        plt.title(label="Frequency spectrum (DFT)")
        plt.xlabel(xlabel="Frequency [Hz]")
        plt.ylabel(ylabel="Amplitude")
        plt.bar(x=self._dft.usable_frequencies, height=self._dft.amplitude, width=1)
        plt.show()
    #-------------------------------------------------------------------------
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def main() -> None:
    signal_duration : float = 0.875 * 5 # initial duration set to 0.875 but decided to lengthen by multiplying by 5
    signal_num_samples : int = 8 * 5 * 32 # originall 8 sambles for a duration of 0.875, once the signal was lengthen by 5 wr then multiplied by 5 and then later we multiplied by 32 to add more data
    signal: Signal = Signal(duration=signal_duration, num_samples=signal_num_samples)
    dft: DiscreteFourierTransform = DiscreteFourierTransform(signal=signal)
    plotter: Plotter = Plotter(signal=signal, dft=dft)

    plotter.plot_time_domain()
    plotter.plot_frequency_spectrum()
#-------------------------------------------------------------------------

if __name__ == "__main__":
    main()
