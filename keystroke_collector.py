import keyboard
import json
import time
import os
from datetime import datetime

class KeystrokeCollector:
    def __init__(self, output_dir="keystroke_data"):
        self.output_dir = output_dir
        self.current_session = []
        self.last_keystroke_time = None
        self.session_timeout = 2.0
        self.user_id = None
        self.session_id = None
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if not os.path.exists(f"{output_dir}/raw"):
            os.makedirs(f"{output_dir}/raw")
        if not os.path.exists(f"{output_dir}/processed"):
            os.makedirs(f"{output_dir}/processed")
    
    def start_collection(self, user_id="unknown"):
        self.user_id = user_id
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Starting keystroke collection for user: {user_id}")
        print("Press Ctrl+C to stop collection")
        
        try:
            keyboard.hook(self._keystroke_callback)
            keyboard.wait('ctrl+c')
        except KeyboardInterrupt:
            pass
        finally:
            keyboard.unhook_all()
            if self.current_session:
                self._save_current_session()
            print(f"Keystroke collection stopped for user: {user_id}")
    
    def _keystroke_callback(self, event):
        current_time = time.time()
        
        if self.last_keystroke_time is not None and \
           (current_time - self.last_keystroke_time) > self.session_timeout:
            self._save_current_session()
            self.current_session = []
        
        self.last_keystroke_time = current_time
        
        self.current_session.append({
            "event_type": event.event_type,
            "scan_code": event.scan_code,
            "name": event.name,
            "time": event.time,
            "is_keypad": event.is_keypad
        })
    
    def _save_current_session(self):
        if len(self.current_session) < 10:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/raw/{self.user_id}_{self.session_id}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.current_session, f, indent=2)
        
        print(f"Saved session with {len(self.current_session)} keystrokes to {filename}")

if __name__ == "__main__":
    user_id = input("Enter user ID: ")
    collector = KeystrokeCollector()
    collector.start_collection(user_id) 