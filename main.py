import os
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks
import time
import threading
from collections import deque

# Adjustable Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_WINDOW_SECONDS = 10  # 10 seconds for analysis window
UPDATE_INTERVAL_SECONDS = 1  # Update display every 1 second
LOWCUT = 9000  # Lower bound of the frequency range in Hz
HIGHCUT = 15000  # Upper bound of the frequency range in Hz
GAIN = 30  # Gain for amplification
INITIAL_PEAK_THRESHOLD = 3.5  # Initial threshold for peak detection
TRIM_THRESHOLD = 0.01  # Threshold for trimming audio
COMMON_BPH_VALUES = [18000, 19800, 21600, 25200, 28800, 36000]  # Common BPH values in watches

# Global variables for continuous monitoring
running = False
accuracy_history = deque(maxlen=100)  # Store last 100 accuracy measurements
current_bph = None
audio_buffer = deque(maxlen=int(RATE * RECORD_WINDOW_SECONDS / CHUNK))  # Buffer for 10 seconds of audio

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

# Record audio continuously and maintain buffer
def record_audio_continuous(device_index, rate=RATE, chunk=CHUNK):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=rate,
                        input=True,
                        input_device_index=device_index,
                        frames_per_buffer=chunk)

    try:
        while running:
            data = stream.read(chunk)
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_buffer.append(audio_data)
    except Exception as e:
        print(f"Error in continuous recording: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

# Get the current audio buffer as a single array
def get_current_audio_buffer():
    if len(audio_buffer) == 0:
        return None
    return np.hstack(list(audio_buffer))

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
def analyze_ticks(peaks, rate, trimmed_audio):
    if len(peaks) < 2:
        return None, None, None, None, None, None

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

# Calculate daily accuracy
def calculate_daily_accuracy(intervals, bph):
    if intervals is None or len(intervals) == 0:
        return None

    # Calculate expected interval
    expected_interval = 1 / (bph / 3600)  # Convert BPH to beats per second

    # Calculate accuracy
    accuracy = (intervals - expected_interval) * 1000  # Convert to milliseconds
    daily_accuracy_seconds = (np.mean(accuracy) / 1000) * 24 * 3600

    return daily_accuracy_seconds, np.mean(accuracy)

# Live monitoring function with sliding window
def live_monitor(device_index):
    global running, accuracy_history, current_bph

    print(f"\nStarting live monitoring with BPH: {current_bph}")
    print(f"Recording window: {RECORD_WINDOW_SECONDS} seconds")
    print(f"Update interval: {UPDATE_INTERVAL_SECONDS} second")
    print("Press Ctrl+C to stop monitoring...")
    print("=" * 50)

    # Start continuous recording in a separate thread
    recording_thread = threading.Thread(target=record_audio_continuous, args=(device_index,))
    recording_thread.daemon = True
    recording_thread.start()

    # Wait for buffer to fill up initially
    print("Filling audio buffer...")
    time.sleep(RECORD_WINDOW_SECONDS)

    last_update_time = time.time()

    while running:
        current_time = time.time()

        # Update every UPDATE_INTERVAL_SECONDS
        if current_time - last_update_time >= UPDATE_INTERVAL_SECONDS:
            try:
                # Get current audio buffer
                audio_data = get_current_audio_buffer()

                if audio_data is not None and len(audio_data) > 0:
                    # Normalize audio data
                    audio_data = audio_data / np.max(np.abs(audio_data))

                    # Apply bandpass filter
                    filtered_audio = bandpass_filter(audio_data, LOWCUT, HIGHCUT, RATE)

                    # Increase gain
                    filtered_audio *= GAIN

                    # Trim audio
                    trimmed_audio = trim_audio(filtered_audio)

                    # Detect peaks
                    peaks, properties = find_peaks(trimmed_audio, height=INITIAL_PEAK_THRESHOLD, distance=RATE/10)

                    if len(peaks) >= 2:
                        # Analyze ticks
                        result = analyze_ticks(peaks, RATE, trimmed_audio)
                        if result[0] is not None:
                            intervals, beat_errors, average_beat_error, tick_frequency, amplitudes, amplitude_variance = result

                            # Calculate accuracy
                            accuracy_result = calculate_daily_accuracy(intervals, current_bph)
                            if accuracy_result is not None:
                                daily_accuracy, avg_accuracy_ms = accuracy_result
                                accuracy_history.append(daily_accuracy)

                                # Clear screen and display results
                                clear_terminal()
                                print(f"Live Watch Accuracy Monitor - BPH: {current_bph}")
                                print(f"Analysis Window: {RECORD_WINDOW_SECONDS} seconds | Update: {UPDATE_INTERVAL_SECONDS} second")
                                print("=" * 60)
                                print(f"Current Daily Accuracy: {daily_accuracy:.2f} seconds")
                                print(f"Average Tick Accuracy: {avg_accuracy_ms:.2f} ms")
                                print(f"Average Beat Error: {average_beat_error:.4f} s")
                                print(f"Tick Frequency: {tick_frequency:.2f} Hz")
                                print(f"Amplitude Variance: {amplitude_variance:.4f}")
                                print(f"Ticks Detected: {len(peaks)}")
                                print(f"Buffer Size: {len(audio_buffer)} chunks")

                                if len(accuracy_history) > 1:
                                    avg_daily_accuracy = np.mean(accuracy_history)
                                    print(f"Average Daily Accuracy (last {len(accuracy_history)} readings): {avg_daily_accuracy:.2f} seconds")

                                print("=" * 60)
                                print("Monitoring... (Press Ctrl+C to stop)")
                            else:
                                print("No valid ticks detected in this sample")
                        else:
                            print("Insufficient ticks detected for analysis")
                    else:
                        print("No peaks detected in this sample")

                else:
                    print("No audio data available")

            except Exception as e:
                print(f"Error during monitoring: {e}")

            last_update_time = current_time

        time.sleep(0.1)  # Small delay to prevent excessive CPU usage

# Main script
if __name__ == "__main__":
    clear_terminal()
    print("Watch Accuracy Monitor")
    print("=" * 30)

    # Ask for BPH at the beginning
    print("Please enter the BPH (Beats Per Hour) for your watch:")
    print("Common BPH values:", COMMON_BPH_VALUES)

    while True:
        try:
            bph_input = input("Enter BPH value: ").strip()
            if bph_input.lower() == 'auto':
                # Auto-detect BPH
                list_input_devices()
                device_index = int(input("Enter the index of the device you want to use: "))

                print("Recording sample to estimate BPH...")
                # Use a simple recording for BPH estimation
                audio = pyaudio.PyAudio()
                stream = audio.open(format=FORMAT,
                                    channels=CHANNELS,
                                    rate=RATE,
                                    input=True,
                                    input_device_index=device_index,
                                    frames_per_buffer=CHUNK)

                frames = []
                for _ in range(0, int(RATE / CHUNK * 5)):  # Record 5 seconds for estimation
                    data = stream.read(CHUNK)
                    frames.append(np.frombuffer(data, dtype=np.int16))

                stream.stop_stream()
                stream.close()
                audio.terminate()

                audio_data = np.hstack(frames)
                audio_data = audio_data / np.max(np.abs(audio_data))
                filtered_audio = bandpass_filter(audio_data, LOWCUT, HIGHCUT, RATE)
                filtered_audio *= GAIN
                trimmed_audio = trim_audio(filtered_audio)
                peaks, _ = find_peaks(trimmed_audio, height=INITIAL_PEAK_THRESHOLD, distance=RATE/10)

                if len(peaks) >= 2:
                    intervals = np.diff(peaks) / RATE
                    estimated_bph = estimate_bph(intervals)
                    closest_bph = closest_standard_bph(estimated_bph)
                    print(f"Estimated BPH: {closest_bph}")
                    current_bph = closest_bph
                else:
                    print("Could not detect ticks for BPH estimation. Please enter manually.")
                    continue
            else:
                current_bph = int(bph_input)
            break
        except ValueError:
            print("Please enter a valid number or 'auto' for automatic detection")

    # List devices and get device index
    list_input_devices()
    device_index = int(input("Enter the index of the device you want to use: "))

    # Start live monitoring
    running = True
    try:
        live_monitor(device_index)
    except KeyboardInterrupt:
        running = False
        print("\nMonitoring stopped.")
        print("Final statistics:")
        if len(accuracy_history) > 0:
            print(f"Average daily accuracy: {np.mean(accuracy_history):.2f} seconds")
            print(f"Min daily accuracy: {np.min(accuracy_history):.2f} seconds")
            print(f"Max daily accuracy: {np.max(accuracy_history):.2f} seconds")
