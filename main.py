import os
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks

# Adjustable Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 20
LOWCUT = 9109  # Lower bound of the frequency range in Hz
HIGHCUT = 14452  # Upper bound of the frequency range in Hz
GAIN = 20  # Gain for amplification
INITIAL_PEAK_THRESHOLD = 0.15  # Initial threshold for peak detection
TRIM_THRESHOLD = 0.01  # Threshold for trimming audio
COMMON_BPH_VALUES = [18000, 19800, 21600, 25200, 28800, 36000]  # Common BPH values in watches

def clear_terminal():
    """
    Clears the terminal screen for Windows, macOS, and Linux.
    """
    if os.name == 'nt':  # For Windows
        os.system('cls')
    else:  # For macOS and Linux
        os.system('clear')

# List available input devices
def list_input_devices():
    audio = pyaudio.PyAudio()
    print("Available audio input devices:")
    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            print(f"{i}- {device_info['name']}")
    audio.terminate()

# Record audio
def record_audio(device_index, record_seconds=RECORD_SECONDS, rate=RATE, chunk=CHUNK):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=chunk)

    print(f"Recording for {RECORD_SECONDS} seconds...")
    frames = []
    for _ in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    return np.hstack(frames)

# Bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Trim audio
def trim_audio(data, threshold=TRIM_THRESHOLD):
    start = 0
    end = len(data)

    # Find start
    for i in range(len(data)):
        if np.abs(data[i]) > threshold:
            start = i
            break

    # Find end
    for i in range(len(data)-1, -1, -1):
        if np.abs(data[i]) > threshold:
            end = i
            break

    return data[start:end]

# Analyze ticks
def analyze_ticks(peaks, rate):
    # Calculate intervals between ticks
    intervals = np.diff(peaks) / rate  # Convert from samples to seconds

    # Calculate beat error
    beat_errors = np.abs(np.diff(intervals))
    average_beat_error = np.mean(beat_errors)

    # Calculate frequency of ticks
    tick_frequency = 1 / np.mean(intervals)

    # Calculate amplitude of ticks
    amplitudes = trimmed_audio[peaks]
    amplitude_variance = np.var(amplitudes)

    return intervals, beat_errors, average_beat_error, tick_frequency, amplitudes, amplitude_variance

# Estimate Beats Per Hour (BPH)
def estimate_bph(intervals):
    average_interval = np.mean(intervals)  # in seconds
    estimated_bph = (1 / average_interval) * 3600  # Convert to beats per hour
    return estimated_bph

# Find the closest standard BPH value
def closest_standard_bph(estimated_bph, common_bph_values=COMMON_BPH_VALUES):
    closest_bph = min(common_bph_values, key=lambda x: abs(x - estimated_bph))
    return closest_bph

# Main script
if __name__ == "__main__":
    clear_terminal()
    list_input_devices()
    device_index = int(input("Enter the index of the device you want to use: "))
    clear_terminal()

    audio_data = record_audio(device_index)

    # Normalize audio data
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Apply bandpass filter
    filtered_audio = bandpass_filter(audio_data, LOWCUT, HIGHCUT, RATE)

    # Increase gain
    filtered_audio *= GAIN

    # Trim audio
    trimmed_audio = trim_audio(filtered_audio)

    # Define threshold for peak detection
    peak_threshold = INITIAL_PEAK_THRESHOLD

    # Detect peaks in the trimmed audio signal
    peaks, properties = find_peaks(trimmed_audio, height=peak_threshold, distance=RATE/10)

    # Estimate BPH
    intervals = np.diff(peaks) / RATE
    estimated_bph = estimate_bph(intervals)
    closest_bph = closest_standard_bph(estimated_bph)

    # Ask user if they want to use the estimated BPH or manually enter BPH
    use_auto_bph = input(f"Estimated BPH is {closest_bph}. Do you want to use this value? (y/n): ").strip().lower()

    if use_auto_bph == 'y':
        BPH = closest_bph
    else:
        BPH = int(input("Enter the BPH value manually: "))

    # Analyze ticks
    intervals, beat_errors, average_beat_error, tick_frequency, amplitudes, amplitude_variance = analyze_ticks(peaks, RATE)

    # Calculate expected interval
    expected_interval = 1 / (BPH / 3600)  # Convert BPH to beats per second

    # Calculate accuracy
    accuracy = (intervals - expected_interval) * 1000  # Convert to milliseconds
    daily_accuracy_seconds = ( ( np.sum(accuracy) / len(accuracy) ) / 1000 ) * 24 * 3600

    # Output analysis results
    print("=======================")
    print("Average beat error (s):", average_beat_error)
    print("Tick frequency (Hz):", tick_frequency)
    print("Amplitude variance:", amplitude_variance)
    print("Average tick accuracy (ms):", np.mean(accuracy))
    print("Daily accuracy (s): ",daily_accuracy_seconds)
    print("=======================")

    # Plot the audio signal and detected peaks
    plt.figure(figsize=(12, 6))
    plt.plot(trimmed_audio, label='Filtered and Trimmed Audio Signal')
    plt.plot(peaks, trimmed_audio[peaks], "x", label='Detected Peaks')
    plt.legend()
    plt.title("Filtered and Trimmed Audio Signal with Detected Peaks")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.show()

    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(accuracy, label='Accuracy (ms)')
    plt.hlines(0, 0, len(accuracy), colors='r', linestyles='dashed', label='Expected Interval')
    plt.legend()
    plt.title("Watch Accuracy Over Time")
    plt.xlabel("Tick Number")
    plt.ylabel("Accuracy (ms)")
    plt.show()
