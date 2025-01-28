from pylsl import StreamInlet, resolve_byprop
from scipy.signal import lfilter, lfilter_zi
from mne.filter import create_filter
import numpy as np
import time
import keyboard

LSL_SCAN_TIMEOUT = 5
LSL_EEG_CHUNK = 12

def acquire_and_filter_eeg():
    print("Looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=LSL_SCAN_TIMEOUT)

    if len(streams) == 0:
        raise RuntimeError("Can't find EEG stream.")
    print("Start acquiring data.")

    # Connect to the EEG stream
    inlet = StreamInlet(streams[0], max_chunklen=LSL_EEG_CHUNK)
    info = inlet.info()

    # Get the sampling frequency and number of channels
    sfreq = info.nominal_srate()
    n_chans = info.channel_count()

    # Create a bandpass filter
    bf = create_filter(np.zeros((int(sfreq), n_chans)), sfreq, 3, 40., method='fir')
    af = [1.0]
    zi = lfilter_zi(bf, af)
    filt_state = np.tile(zi, (n_chans, 1)).transpose()

    print("Filtering EEG data... Press Ctrl+C to stop.")
    try:
        while True:
            # Pull EEG data in chunks
            samples, timestamps = inlet.pull_chunk(timeout=1.0, max_samples=100)
            if timestamps:
                samples = np.array(samples)[:, ::-1]

                # Apply the filter
                filt_samples, filt_state = lfilter(bf, af, samples, axis=0, zi=filt_state)

                # Display the filtered samples
                print("Filtered EEG samples:", filt_samples[0][1:5])
                if filt_samples[0][1:5][0] > 40 or filt_samples[0][1:5][-1]>40:
                    print("Blink detected!")
                    keyboard.press('space')
                    time.sleep(0.1)  # Attente courte pour simuler la durée de la pression
                    keyboard.release('space')
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Acquisition stopped.")

if __name__ == "__main__":
    acquire_and_filter_eeg()
