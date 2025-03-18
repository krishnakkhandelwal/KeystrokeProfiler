import pandas as pd
import numpy as np
import pickle
import os
import json
import time
import keyboard
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.impute import SimpleImputer
import tkinter as tk
from threading import Thread

class KeystrokeIdentifier:
    def __init__(self, dataset_path=None):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
        if dataset_path:
            self.train_model(dataset_path)
    
    def train_model(self, dataset_path):
        print("Training model on dataset:", dataset_path)
        
        data = pd.read_csv(dataset_path)
        print(f"Dataset shape: {data.shape}")
        print(f"Number of users: {data['user_id'].nunique()}")
        
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing values in dataset:")
            print(missing_values[missing_values > 0])
        
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        X = data[numeric_cols].drop(columns=['user_id'] if 'user_id' in numeric_cols else [])
        y = data['user_id']
        
        self.feature_names = X.columns.tolist()
        
        self.imputer = SimpleImputer(strategy='mean')
        X_imputed = self.imputer.fit_transform(X)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        
        n_samples = len(data)
        if n_samples < 5 or min_class_count < 3:
            print(f"Small dataset detected ({n_samples} samples, minimum {min_class_count} per class).")
            print("Using Leave-One-Out cross-validation.")
            cv_strategy = LeaveOneOut()
        else:
            cv_folds = min(5, min_class_count)
            print(f"Using {cv_folds}-fold cross-validation.")
            cv_strategy = cv_folds
        
        if n_samples < 4 or min_class_count < 2:
            print("Dataset too small for train/test split. Using all data for training.")
            X_train, y_train = X_scaled, y
            X_test, y_test = X_scaled, y
        else:
            test_size = 0.2 if n_samples >= 10 else 0.25
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, 
                stratify=y if len(y.unique()) > 1 and all(class_counts > 1) else None
            )
        
        models = {
            'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
        }
        
        best_score = 0
        best_model = None
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                
                test_score = model.score(X_test, y_test)
                print(f"{name} Test Accuracy: {test_score:.4f}")
                
                cv_scores = cross_val_score(model, X_scaled, y, cv=cv_strategy)
                print(f"{name} CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
                if test_score > best_score:
                    best_score = test_score
                    best_model = model
            except Exception as e:
                print(f"Error training {name} model: {e}")
                
                if name == 'Neural Network':
                    try:
                        print("Trying simpler neural network...")
                        simple_nn = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500, random_state=42)
                        simple_nn.fit(X_train, y_train)
                        test_score = simple_nn.score(X_test, y_test)
                        print(f"Simple Neural Network Test Accuracy: {test_score:.4f}")
                        
                        if test_score > best_score:
                            best_score = test_score
                            best_model = simple_nn
                    except Exception as e2:
                        print(f"Error training simple neural network: {e2}")
        
        if best_model is None:
            print("No model was successfully trained. Using a simple classifier.")
            best_model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
            best_model.fit(X_train, y_train)
            best_score = best_model.score(X_test, y_test)
        
        self.model = best_model
        print(f"Selected model with test accuracy: {best_score:.4f}")
        
        with open('keystroke_model.pkl', 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'imputer': self.imputer,
                'feature_names': self.feature_names
            }, f)
        
        print("Model saved to keystroke_model.pkl")
        return best_score
    
    def load_model(self, model_path='keystroke_model.pkl'):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.imputer = model_data.get('imputer')
        print("Model loaded successfully")
    
    def identify_user(self, keystroke_data):
        features = self._extract_features(keystroke_data)
        
        features_df = pd.DataFrame([features], columns=self.feature_names)
        
        if self.imputer:
            features_imputed = self.imputer.transform(features_df)
        else:
            features_imputed = features_df.fillna(features_df.mean()).values
        
        scaled_features = self.scaler.transform(features_imputed)
        
        user_id = self.model.predict(scaled_features)[0]
        probabilities = self.model.predict_proba(scaled_features)[0]
        
        confidence_scores = {}
        for i, prob in enumerate(probabilities):
            user = self.model.classes_[i]
            confidence_scores[user] = float(prob)
        
        return user_id, confidence_scores
    
    def _extract_features(self, keystroke_data):
        key_down_times = {}
        
        hold_times = []
        down_down_times = []
        up_down_times = []
        
        shift_count = 0
        backspace_count = 0
        special_key_count = 0
        
        last_down_time = None
        last_up_time = None
        
        for event in keystroke_data:
            key = event['name']
            event_type = event['event_type']
            event_time = event['time']
            
            if key in ['shift', 'right shift', 'left shift']:
                shift_count += 1
            elif key == 'backspace':
                backspace_count += 1
            elif key in ['ctrl', 'alt', 'tab', 'esc', 'caps lock', 'enter']:
                special_key_count += 1
            
            if event_type == 'down':
                key_down_times[key] = event_time
                
                if last_down_time is not None:
                    down_down_times.append(event_time - last_down_time)
                last_down_time = event_time
                
                if last_up_time is not None:
                    up_down_times.append(event_time - last_up_time)
                
            elif event_type == 'up':
                if key in key_down_times:
                    hold_time = event_time - key_down_times[key]
                    hold_times.append(hold_time)
                    del key_down_times[key]
                
                last_up_time = event_time
        
        features = {}
        
        if hold_times:
            features['mean_hold_time'] = np.mean(hold_times)
            features['std_hold_time'] = np.std(hold_times)
            features['min_hold_time'] = np.min(hold_times)
            features['max_hold_time'] = np.max(hold_times)
            features['median_hold_time'] = np.median(hold_times)
            
            features['hold_time_25th'] = np.percentile(hold_times, 25)
            features['hold_time_75th'] = np.percentile(hold_times, 75)
        
        if down_down_times:
            features['mean_down_down_time'] = np.mean(down_down_times)
            features['std_down_down_time'] = np.std(down_down_times)
            features['min_down_down_time'] = np.min(down_down_times)
            features['max_down_down_time'] = np.max(down_down_times)
            features['median_down_down_time'] = np.median(down_down_times)
            
            features['down_down_25th'] = np.percentile(down_down_times, 25)
            features['down_down_75th'] = np.percentile(down_down_times, 75)
        
        if up_down_times:
            features['mean_up_down_time'] = np.mean(up_down_times)
            features['std_up_down_time'] = np.std(up_down_times)
            features['min_up_down_time'] = np.min(up_down_times)
            features['max_up_down_time'] = np.max(up_down_times)
        
        features['shift_frequency'] = shift_count / len(keystroke_data) if len(keystroke_data) > 0 else 0
        features['backspace_frequency'] = backspace_count / len(keystroke_data) if len(keystroke_data) > 0 else 0
        features['special_key_frequency'] = special_key_count / len(keystroke_data) if len(keystroke_data) > 0 else 0
        
        if len(keystroke_data) >= 2:
            typing_duration = keystroke_data[-1]['time'] - keystroke_data[0]['time']
            features['keystrokes_per_second'] = len(keystroke_data) / typing_duration if typing_duration > 0 else 0
        
        features['total_keystrokes'] = len(keystroke_data)
        
        if len(down_down_times) > 1:
            features['rhythm_consistency'] = 1.0 / (np.std(down_down_times) + 0.0001)
        
        return features

class RealTimeIdentifier:
    def __init__(self):
        self.identifier = KeystrokeIdentifier()
        self.identifier.load_model()
        
        self.keystroke_window = []
        self.window_size = 20
        self.min_keystrokes = 10
        self.running = True
        
        self.root = tk.Tk()
        self.root.title("Keystroke Dynamics Identifier")
        self.root.geometry("400x500")
        
        self.status_label = tk.Label(self.root, text="Monitoring keystrokes...", font=("Arial", 12))
        self.status_label.pack(pady=10)
        
        self.current_user_label = tk.Label(self.root, text="Current User: Unknown", font=("Arial", 14, "bold"))
        self.current_user_label.pack(pady=10)
        
        self.confidence_frame = tk.Frame(self.root)
        self.confidence_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.keystroke_count_label = tk.Label(self.root, text="Keystrokes: 0", font=("Arial", 10))
        self.keystroke_count_label.pack(pady=5)
        
        self.reset_button = tk.Button(self.root, text="Reset", command=self._reset_window)
        self.reset_button.pack(pady=10)
        
        self.keyboard_thread = Thread(target=self._start_monitoring)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()
    
    def _reset_window(self):
        self.keystroke_window = []
        self.keystroke_count_label.config(text=f"Keystrokes: 0")
        self.current_user_label.config(text="Current User: Unknown")
        for widget in self.confidence_frame.winfo_children():
            widget.destroy()
    
    def _start_monitoring(self):
        keyboard.hook(self._keystroke_callback)
    
    def _keystroke_callback(self, event):
        if event.name == 'esc' and event.event_type == 'down':
            return
        
        self.keystroke_window.append({
            "event_type": event.event_type,
            "scan_code": event.scan_code,
            "name": event.name,
            "time": event.time,
            "is_keypad": event.is_keypad
        })
        
        if len(self.keystroke_window) > self.window_size:
            self.keystroke_window = self.keystroke_window[-self.window_size:]
        
        self.keystroke_count_label.config(text=f"Keystrokes: {len(self.keystroke_window)}")
        
        if len(self.keystroke_window) >= self.min_keystrokes:
            user_id, confidence_scores = self.identifier.identify_user(self.keystroke_window)
            self._update_ui(user_id, confidence_scores)
    
    def _update_ui(self, user_id, confidence_scores):
        self.current_user_label.config(text=f"Current User: {user_id}")
        
        for widget in self.confidence_frame.winfo_children():
            widget.destroy()
        
        sorted_users = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (user, confidence) in enumerate(sorted_users):
            font = ("Arial", 10, "bold") if user == user_id else ("Arial", 10)
            
            bg_color = "#e6ffe6" if confidence > 0.7 else "#ffffff"
            
            label = tk.Label(
                self.confidence_frame, 
                text=f"{user}: {confidence:.2f}",
                font=font,
                anchor='w',
                bg=bg_color,
                width=30,
                padx=10
            )
            label.pack(fill='x', padx=5, pady=2)
    
    def start(self):
        self.root.mainloop()
        self.running = False
        keyboard.unhook_all()

if __name__ == "__main__":
    app = RealTimeIdentifier()
    app.start() 