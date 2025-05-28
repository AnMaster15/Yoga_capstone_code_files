import pickle
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import neurokit2 as nk
import os
import matplotlib.pyplot as plt
import time  # Add this import
import warnings  # Add this import

# Suppress warnings globally for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_PATH = "WESAD-2"

def load_subject_data(subject_id):
    """Load synchronized data from .pkl file for a given subject"""
    file_path = os.path.join(BASE_PATH, f"S{subject_id}", f"S{subject_id}.pkl")
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')  # Use latin1 for Python 3 compatibility
    return data

def extract_signals(data):
    """Extract relevant signals from the data dictionary"""
    # Extract signals from chest device (RespiBAN)
    ecg = data['signal']['chest']['ECG'].flatten()
    emg = data['signal']['chest']['EMG'].flatten()
    eda_chest = data['signal']['chest']['EDA'].flatten()
    
    # Extract signals from wrist device (E4)
    bvp = data['signal']['wrist']['BVP'].flatten()
    eda_wrist = data['signal']['wrist']['EDA'].flatten()
    
    # Extract labels (0=not defined, 1=baseline, 2=stress, 3=amusement)
    labels = data['label']
    
    return ecg, emg, eda_chest, bvp, eda_wrist, labels

def extract_ecg_features(ecg_signal, sampling_rate=700):
    """Extract features from ECG signal"""
    try:
        # Process ECG signal using neurokit2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ecg_signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
        
        # Calculate heart rate
        heart_rate = np.nan
        if 'ECG_R_Peaks' in info and len(info['ECG_R_Peaks']) > 1:
            # Calculate average RR interval in milliseconds
            rr_intervals = np.diff(info['ECG_R_Peaks']) / sampling_rate * 1000
            if len(rr_intervals) > 0:
                mean_rr = np.mean(rr_intervals)
                heart_rate = 60000 / mean_rr if mean_rr > 0 else np.nan
                
                # Calculate SDNN and RMSSD manually
                sdnn = np.std(rr_intervals) if len(rr_intervals) > 1 else np.nan
                rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) if len(rr_intervals) > 1 else np.nan
            else:
                sdnn = np.nan
                rmssd = np.nan
        else:
            sdnn = np.nan
            rmssd = np.nan
        
        # Calculate additional statistical features
        ecg_mean = np.mean(ecg_signal)
        ecg_std = np.std(ecg_signal)
        ecg_skewness = skew(ecg_signal)
        ecg_kurtosis = kurtosis(ecg_signal)
        
        features = {
            'ECG_Mean': ecg_mean,
            'ECG_Std': ecg_std,
            'ECG_Skewness': ecg_skewness,
            'ECG_Kurtosis': ecg_kurtosis,
            'Heart_Rate': heart_rate,
            'SDNN': sdnn,
            'RMSSD': rmssd
        }
    except Exception as e:
        print(f"Error extracting ECG features: {e}")
        features = {
            'ECG_Mean': np.mean(ecg_signal) if len(ecg_signal) > 0 else np.nan,
            'ECG_Std': np.std(ecg_signal) if len(ecg_signal) > 0 else np.nan,
            'ECG_Skewness': skew(ecg_signal) if len(ecg_signal) > 0 else np.nan,
            'ECG_Kurtosis': kurtosis(ecg_signal) if len(ecg_signal) > 0 else np.nan,
            'Heart_Rate': np.nan,
            'SDNN': np.nan,
            'RMSSD': np.nan
        }
    
    return features

def extract_emg_features(emg_signal, sampling_rate=700):
    """Extract features from EMG signal"""
    try:
        # Apply a more appropriate bandpass filter for EMG
        # Using values well below the Nyquist frequency (sampling_rate/2)
        nyquist = sampling_rate / 2
        sos = signal.butter(4, [20, min(250, nyquist-1)], 'bandpass', fs=sampling_rate, output='sos')
        filtered_emg = signal.sosfilt(sos, emg_signal)
        
        # Calculate EMG envelope using RMS method instead of Hilbert transform
        window_size = int(sampling_rate * 0.125)  # 125ms window
        emg_envelope = np.zeros_like(filtered_emg)
        for i in range(len(filtered_emg)):
            start = max(0, i - window_size // 2)
            end = min(len(filtered_emg), i + window_size // 2)
            emg_envelope[i] = np.sqrt(np.mean(filtered_emg[start:end]**2))
        
        # Calculate statistical features
        emg_mean = np.mean(emg_envelope)
        emg_std = np.std(emg_envelope)
        emg_max = np.max(emg_envelope)
        emg_skewness = skew(emg_envelope)
        emg_kurtosis = kurtosis(emg_envelope)
        
        features = {
            'EMG_Mean': emg_mean,
            'EMG_Std': emg_std,
            'EMG_Max': emg_max,
            'EMG_Skewness': emg_skewness,
            'EMG_Kurtosis': emg_kurtosis
        }
    except Exception as e:
        print(f"Error extracting EMG features: {e}")
        # Just use basic statistics on the raw signal instead
        features = {
            'EMG_Mean': np.mean(emg_signal) if len(emg_signal) > 0 else np.nan,
            'EMG_Std': np.std(emg_signal) if len(emg_signal) > 0 else np.nan,
            'EMG_Max': np.max(emg_signal) if len(emg_signal) > 0 else np.nan,
            'EMG_Skewness': skew(emg_signal) if len(emg_signal) > 0 else np.nan,
            'EMG_Kurtosis': kurtosis(emg_signal) if len(emg_signal) > 0 else np.nan
        }
    
    return features

def extract_eda_features(eda_signal, sampling_rate=700):
    """Extract features from EDA/GSR signal"""
    try:
        # For low sampling rates, use different approach
        if sampling_rate < 10:  # For signals like the wrist EDA (4Hz)
            # Calculate statistical features directly
            eda_mean = np.mean(eda_signal)
            eda_std = np.std(eda_signal)
            eda_max = np.max(eda_signal)
            
            # Apply moving average smoothing
            window_size = max(3, int(sampling_rate))  # At least 3 samples
            smoothed_eda = np.convolve(eda_signal, np.ones(window_size)/window_size, mode='valid')
            
            # Simplified peak detection for low-frequency signals
            # Using gradient-based approach which is more suitable for low sampling rates
            if len(smoothed_eda) > 2:
                # Calculate first derivative
                eda_diff = np.diff(smoothed_eda)
                # Find where derivative changes from positive to negative (peaks)
                peak_indices = np.where((eda_diff[:-1] > 0) & (eda_diff[1:] < 0))[0] + 1
                scr_count = len(peak_indices)
                
                # Calculate SCR amplitudes
                scr_amplitudes = []
                for peak in peak_indices:
                    if peak + 1 < len(smoothed_eda) and peak > 0:
                        # Look for recovery in the next few samples
                        search_end = min(peak + int(sampling_rate * 5), len(smoothed_eda))
                        if search_end > peak + 1:
                            recovery_idx = peak + np.argmin(smoothed_eda[peak:search_end])
                            if recovery_idx > peak:
                                amplitude = smoothed_eda[peak] - smoothed_eda[recovery_idx]
                                if amplitude > 0:
                                    scr_amplitudes.append(amplitude)
                
                scr_amplitude_mean = np.mean(scr_amplitudes) if len(scr_amplitudes) > 0 else np.nan
            else:
                scr_count = 0
                scr_amplitude_mean = np.nan
        else:
            # For higher sampling rates, use neurokit with appropriate parameters
            # Suppress warnings for cleaner output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eda_signals, info = nk.eda_process(eda_signal, sampling_rate=sampling_rate)
            
            # Extract features
            scr_count = len(info['SCR_Peaks']) if 'SCR_Peaks' in info else 0
            
            # Calculate SCR amplitudes manually if peaks are available
            scr_amplitudes = []
            if 'SCR_Peaks' in info and len(info['SCR_Peaks']) > 0:
                for peak in info['SCR_Peaks']:
                    # Look for recovery point (minimum) after peak
                    search_window = min(int(sampling_rate * 5), len(eda_signal) - peak - 1)  # 5 sec window
                    if search_window > 0:
                        recovery_idx = peak + np.argmin(eda_signal[peak:peak+search_window])
                        if recovery_idx > peak:
                            amplitude = eda_signal[peak] - eda_signal[recovery_idx]
                            if amplitude > 0:
                                scr_amplitudes.append(amplitude)
            
            scr_amplitude_mean = np.mean(scr_amplitudes) if len(scr_amplitudes) > 0 else np.nan
            
            # Calculate statistical features
            eda_mean = np.mean(eda_signal)
            eda_std = np.std(eda_signal)
            eda_max = np.max(eda_signal)
        
        features = {
            'EDA_Mean': eda_mean,
            'EDA_Std': eda_std,
            'EDA_Max': eda_max,
            'SCR_Count': scr_count,
            'SCR_Amplitude_Mean': scr_amplitude_mean
        }
    except Exception as e:
        print(f"Error extracting EDA features: {e}")
        features = {
            'EDA_Mean': np.mean(eda_signal) if len(eda_signal) > 0 else np.nan,
            'EDA_Std': np.std(eda_signal) if len(eda_signal) > 0 else np.nan,
            'EDA_Max': np.max(eda_signal) if len(eda_signal) > 0 else np.nan,
            'SCR_Count': np.nan,
            'SCR_Amplitude_Mean': np.nan
        }
    
    return features

def extract_bvp_features(bvp_signal, sampling_rate=64):
    """Extract features from BVP signal and estimate SpO2"""
    try:
        # Process PPG (BVP) signal using neurokit2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ppg_signals, info = nk.ppg_process(bvp_signal, sampling_rate=sampling_rate)
        
        # Calculate pulse rate manually if peaks are available
        pulse_rate = np.nan
        if 'PPG_Peaks' in info and len(info['PPG_Peaks']) > 1:
            # Calculate average peak-to-peak interval in seconds
            pp_intervals = np.diff(info['PPG_Peaks']) / sampling_rate
            if len(pp_intervals) > 0:
                mean_pp = np.mean(pp_intervals)
                pulse_rate = 60 / mean_pp if mean_pp > 0 else np.nan
        
        # Calculate BVP features
        bvp_mean = np.mean(bvp_signal)
        bvp_std = np.std(bvp_signal)
        bvp_amp = np.max(bvp_signal) - np.min(bvp_signal)
        
        # Simplified SpO2 estimation (this is NOT accurate and just for demonstration)
        if not np.isnan(pulse_rate) and pulse_rate > 0:
        # Create slight variations based on pulse rate
           estimated_spo2 = 98 - abs(pulse_rate - 70) * 0.03
           estimated_spo2 = min(max(estimated_spo2, 94), 99)  # More realistic range
        else:
           estimated_spo2 = 97  # Default value
    
        features = {
        'BVP_Mean': bvp_mean,
        'BVP_Std': bvp_std,
        'BVP_Amplitude': bvp_amp,
        'Pulse_Rate': pulse_rate,
        'Simulated_SpO2': estimated_spo2  # Rename to indicate this is simulated
         }
    except Exception as e:
        print(f"Error extracting BVP features: {e}")
        features = {
            'BVP_Mean': np.mean(bvp_signal) if len(bvp_signal) > 0 else np.nan,
            'BVP_Std': np.std(bvp_signal) if len(bvp_signal) > 0 else np.nan,
            'BVP_Amplitude': (np.max(bvp_signal) - np.min(bvp_signal)) if len(bvp_signal) > 0 else np.nan,
            'Pulse_Rate': np.nan,
            'Estimated_SpO2': 95.0  # Default value
        }
    
    return features

def segment_by_condition(signals, labels, window_size=30, step_size=15, sampling_rate=700):
    """Segment signals by condition with windowing"""
    segments = []
    
    # Map the numerical labels to condition names
    condition_names = {
        0: 'not_defined',
        1: 'baseline',
        2: 'stress', 
        3: 'amusement'
    }
    
    # Get unique conditions (excluding 0 which is not defined)
    unique_conditions = np.unique(labels)
    unique_conditions = unique_conditions[unique_conditions != 0]
    
    for condition in unique_conditions:
        # Find where this condition starts and ends
        condition_mask = (labels == condition)
        condition_indices = np.where(condition_mask)[0]
        
        if len(condition_indices) == 0:
            continue
            
        # Convert indices to samples
        start_sample = condition_indices[0]
        end_sample = condition_indices[-1]
        
        # Create windows
        window_size_samples = int(window_size * sampling_rate)
        step_size_samples = int(step_size * sampling_rate)
        
        for window_start in range(start_sample, end_sample - window_size_samples + 1, step_size_samples):
            window_end = window_start + window_size_samples
            
            # Extract segments for all signals
            ecg_segment = signals[0][window_start:window_end]
            emg_segment = signals[1][window_start:window_end]
            eda_chest_segment = signals[2][window_start:window_end]
            
            # Calculate ratios for wrist signals (different sampling rates)
            bvp_ratio = len(signals[3]) / len(signals[0])
            eda_wrist_ratio = len(signals[4]) / len(signals[0])
            
            bvp_start = int(window_start * bvp_ratio)
            bvp_end = int(window_end * bvp_ratio)
            bvp_segment = signals[3][bvp_start:bvp_end]
            
            eda_wrist_start = int(window_start * eda_wrist_ratio)
            eda_wrist_end = int(window_end * eda_wrist_ratio)
            eda_wrist_segment = signals[4][eda_wrist_start:eda_wrist_end]
            
            # Create segment dictionary
            segment = {
                'start_sample': window_start,
                'end_sample': window_end,
                'condition': condition,
                'condition_name': condition_names.get(condition, 'unknown'),
                'ecg': ecg_segment,
                'emg': emg_segment,
                'eda_chest': eda_chest_segment,
                'bvp': bvp_segment,
                'eda_wrist': eda_wrist_segment
            }
            
            segments.append(segment)
    
    return segments

def process_subject(subject_id):
    """Process all data for a single subject"""
    print(f"Processing subject S{subject_id}...")
    
    try:
        # Load data
        data = load_subject_data(subject_id)
        
        # Extract signals
        ecg, emg, eda_chest, bvp, eda_wrist, labels = extract_signals(data)
        
        # Store original signals
        signals_original = [ecg, emg, eda_chest, bvp, eda_wrist]
        
        # Segment by condition
        segments = segment_by_condition(signals_original, labels)
        
        # Extract features from each segment
        all_features = []
        for i, segment in enumerate(segments):
            if i % 10 == 0:  # Print progress every 10 segments
                print(f"Processing segment {i+1}/{len(segments)}...")
            
            # Extract features from each signal
            ecg_features = extract_ecg_features(segment['ecg'])
            emg_features = extract_emg_features(segment['emg'])
            eda_chest_features = extract_eda_features(segment['eda_chest'])
            
            # Use original sampling rates for these signals
            bvp_features = extract_bvp_features(segment['bvp'], sampling_rate=64)
            eda_wrist_features = extract_eda_features(segment['eda_wrist'], sampling_rate=4)
            
            # Combine all features
            features = {
                'Subject_ID': subject_id,
                'Segment': i,
                'Condition': segment['condition'],
                'Condition_Name': segment['condition_name'],
                'Start_Sample': segment['start_sample'],
                'End_Sample': segment['end_sample'],
                **ecg_features,
                **emg_features,
                **{f"Chest_{k}": v for k, v in eda_chest_features.items()},
                **{f"Wrist_{k}": v for k, v in eda_wrist_features.items()},
                **bvp_features
            }
            
            all_features.append(features)
        
        print(f"Extracted features from {len(all_features)} segments for subject S{subject_id}")
        return all_features
    
    except Exception as e:
        print(f"Error processing subject S{subject_id}: {e}")
        return []

def main():
    """Main function to process multiple subjects and create a merged CSV"""
    # List of subjects to process
    subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    
    # Create a folder for outputs if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each subject and collect features
    all_features = []
    total_subjects = len(subjects)
    
    for i, subject_id in enumerate(subjects):
        print(f"\nProcessing subject S{subject_id} ({i+1}/{total_subjects})...")
        
        # Process the subject
        start_time = time.time()
        subject_features = process_subject(subject_id)
        all_features.extend(subject_features)
        
        # Report completion time
        processing_time = time.time() - start_time
        print(f"Completed processing subject S{subject_id} in {processing_time:.2f} seconds")
        print(f"Extracted {len(subject_features)} feature segments")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'wesad_processed_features2.csv')
    df.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved merged data to {output_file}")
    
    

if __name__ == "__main__":
    main()