import os
import sys
import argparse
from keystroke_collector import KeystrokeCollector
from keystroke_processor import KeystrokeProcessor
from keystroke_analyzer import KeystrokeAnalyzer
from keystroke_identifier import KeystrokeIdentifier, RealTimeIdentifier

def main():
    parser = argparse.ArgumentParser(description="Keystroke Analysis System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    collect_parser = subparsers.add_parser("collect", help="Collect keystroke data")
    collect_parser.add_argument("--user", required=True, help="User ID for data collection")
    
    process_parser = subparsers.add_parser("process", help="Process raw keystroke data")
    
    analyze_parser = subparsers.add_parser("analyze", help="Analyze processed keystroke data")
    
    train_parser = subparsers.add_parser("train", help="Train user identification model")
    
    identify_parser = subparsers.add_parser("identify", help="Run real-time user identification")
    
    args = parser.parse_args()
    
    if args.command == "collect":
        collector = KeystrokeCollector()
        collector.start_collection(args.user)
    
    elif args.command == "process":
        processor = KeystrokeProcessor()
        processor.process_all_files()
        processor.create_dataset()
    
    elif args.command == "analyze":
        dataset_path = "keystroke_data/keystroke_dataset.csv"
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file {dataset_path} not found.")
            print("Run 'process' command first to create the dataset.")
            return
        
        analyzer = KeystrokeAnalyzer(dataset_path)
        analyzer.perform_exploratory_analysis()
        analyzer.perform_data_preprocessing()
        print("Analysis complete. Check the output directory for visualization images.")
    
    elif args.command == "train":
        dataset_path = "keystroke_data/keystroke_dataset.csv"
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file {dataset_path} not found.")
            print("Run 'process' command first to create the dataset.")
            return
        
        identifier = KeystrokeIdentifier(dataset_path)
    
    elif args.command == "identify":
        if not os.path.exists("keystroke_model.pkl"):
            print("Error: Model file not found.")
            print("Run 'train' command first to create the model.")
            return
        
        app = RealTimeIdentifier()
        app.start()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 