import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

class KeystrokeProcessor:
    def __init__(self, data_dir="keystroke_data"):
        self.data_dir = data_dir
        self.raw_dir = f"{data_dir}/raw"
        self.processed_dir = f"{data_dir}/processed"
        
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def process_all_files(self):
        """Process all raw keystroke files"""
        raw_files = [f for f in os.listdir(self.raw_dir) if f.endswith('.json')]
        
        for file in raw_files:
            user_id = file.split('_')[0]
            self.process_file(f"{self.raw_dir}/{file}", user_id)
    
    def process_file(self, file_path, user_id):
        """Process a single raw keystroke file and extract features"""
        print(f"Processing {file_path}...")
        
        # Load raw data
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        
        if len(raw_data) < 20:  # Skip files with too few keystrokes
            print(f"Skipping {file_path} - too few keystrokes")
            return
        
        # Process the entire session as one data point
        features = self._extract_features(raw_data)
        features['user_id'] = user_id
        
        # Save processed data for the full session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{self.processed_dir}/{user_id}_{timestamp}_full_processed.csv"
        pd.DataFrame([features]).to_csv(output_file, index=False)
        
        # Now break the session into chunks to create more data points
        chunk_size = 30  # Number of keystrokes per chunk
        overlap = 15     # Overlap between chunks to increase data points
        
        if len(raw_data) >= chunk_size:
            chunk_count = 0
            for i in range(0, len(raw_data) - chunk_size + 1, chunk_size - overlap):
                chunk = raw_data[i:i + chunk_size]
                
                # Extract features from the chunk
                chunk_features = self._extract_features(chunk)
                chunk_features['user_id'] = user_id
                chunk_features['is_chunk'] = True
                chunk_features['parent_file'] = os.path.basename(file_path)
                
                # Save processed chunk data
                chunk_output_file = f"{self.processed_dir}/{user_id}_{timestamp}_chunk{chunk_count}_processed.csv"
                pd.DataFrame([chunk_features]).to_csv(chunk_output_file, index=False)
                chunk_count += 1
            
            print(f"Created {chunk_count} additional data points from chunks")
        
        print(f"Saved processed data to {output_file}")
    
    def _extract_features(self, raw_data):
        """Extract typing pattern features from raw keystroke data"""
        # Create dictionaries to track key down and up events
        key_down_times = {}
        
        # Lists to store timing data
        hold_times = []  # How long keys are held
        down_down_times = []  # Time between consecutive key presses
        up_down_times = []  # Time between key release and next key press
        
        # Track special key usage
        shift_count = 0
        backspace_count = 0
        special_key_count = 0
        
        last_down_time = None
        last_up_time = None
        
        for event in raw_data:
            key = event['name']
            event_type = event['event_type']
            event_time = event['time']
            
            # Count special keys
            if key in ['shift', 'right shift', 'left shift']:
                shift_count += 1
            elif key == 'backspace':
                backspace_count += 1
            elif key in ['ctrl', 'alt', 'tab', 'esc', 'caps lock', 'enter']:
                special_key_count += 1
            
            if event_type == 'down':
                # Track when the key was pressed down
                key_down_times[key] = event_time
                
                # Calculate time between consecutive key presses
                if last_down_time is not None:
                    down_down_times.append(event_time - last_down_time)
                last_down_time = event_time
                
                # Calculate time between key release and next key press
                if last_up_time is not None:
                    up_down_times.append(event_time - last_up_time)
                
            elif event_type == 'up':
                # Calculate how long the key was held
                if key in key_down_times:
                    hold_time = event_time - key_down_times[key]
                    hold_times.append(hold_time)
                    del key_down_times[key]
                
                last_up_time = event_time
        
        # Calculate statistical features
        features = {}
        
        # Basic statistics for hold times
        if hold_times:
            features['mean_hold_time'] = np.mean(hold_times)
            features['std_hold_time'] = np.std(hold_times)
            features['min_hold_time'] = np.min(hold_times)
            features['max_hold_time'] = np.max(hold_times)
            features['median_hold_time'] = np.median(hold_times)
            
            # Add percentiles for more granularity
            features['hold_time_25th'] = np.percentile(hold_times, 25)
            features['hold_time_75th'] = np.percentile(hold_times, 75)
        
        # Basic statistics for down-down times
        if down_down_times:
            features['mean_down_down_time'] = np.mean(down_down_times)
            features['std_down_down_time'] = np.std(down_down_times)
            features['min_down_down_time'] = np.min(down_down_times)
            features['max_down_down_time'] = np.max(down_down_times)
            features['median_down_down_time'] = np.median(down_down_times)
            
            # Add percentiles
            features['down_down_25th'] = np.percentile(down_down_times, 25)
            features['down_down_75th'] = np.percentile(down_down_times, 75)
        
        # Basic statistics for up-down times
        if up_down_times:
            features['mean_up_down_time'] = np.mean(up_down_times)
            features['std_up_down_time'] = np.std(up_down_times)
            features['min_up_down_time'] = np.min(up_down_times)
            features['max_up_down_time'] = np.max(up_down_times)
        
        # Special key usage features
        features['shift_frequency'] = shift_count / len(raw_data) if len(raw_data) > 0 else 0
        features['backspace_frequency'] = backspace_count / len(raw_data) if len(raw_data) > 0 else 0
        features['special_key_frequency'] = special_key_count / len(raw_data) if len(raw_data) > 0 else 0
        
        # Typing speed features
        if len(raw_data) >= 2:
            typing_duration = raw_data[-1]['time'] - raw_data[0]['time']
            features['keystrokes_per_second'] = len(raw_data) / typing_duration if typing_duration > 0 else 0
        
        # Count total keystrokes
        features['total_keystrokes'] = len(raw_data)
        
        # Calculate rhythm consistency (variance in typing speed)
        if len(down_down_times) > 1:
            features['rhythm_consistency'] = 1.0 / (np.std(down_down_times) + 0.0001)  # Avoid division by zero
        
        return features

    def create_dataset(self, output_file="keystroke_dataset.csv"):
        """Combine all processed files into a single dataset"""
        processed_files = [f for f in os.listdir(self.processed_dir) if f.endswith('.csv')]
        
        if not processed_files:
            print("No processed files found.")
            return
        
        # Combine all files
        all_data = []
        for file in processed_files:
            df = pd.read_csv(f"{self.processed_dir}/{file}")
            all_data.append(df)
        
        # Concatenate into a single DataFrame
        dataset = pd.concat(all_data, ignore_index=True)
        
        # Drop any columns that are all NaN
        dataset = dataset.dropna(axis=1, how='all')
        
        # Fill remaining NaN values with column means
        for col in dataset.columns:
            if col != 'user_id' and dataset[col].dtype != object and dataset[col].isna().any():
                col_mean = dataset[col].mean()
                dataset[col] = dataset[col].fillna(col_mean)
        
        # Save complete dataset
        dataset.to_csv(f"{self.data_dir}/{output_file}", index=False)
        print(f"Created complete dataset: {self.data_dir}/{output_file}")
        print(f"Dataset shape: {dataset.shape}")
        print(f"Number of users: {dataset['user_id'].nunique()}")
        
        return dataset

if __name__ == "__main__":
    processor = KeystrokeProcessor()
    processor.process_all_files()
    processor.create_dataset() 