#!/usr/bin/env python3
"""
EMG UMAP Test Configuration Runner
Allows running umap_test.py with different preset or custom configurations
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path

class UMAPRunner:
    def __init__(self, config_file='gesture_configs.json'):
        self.config_file = config_file
        self.configs = self._load_configs()
    
    def _load_configs(self):
        """Load configuration file"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Config file '{self.config_file}' not found")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in '{self.config_file}'")
            sys.exit(1)
    
    def list_configs(self):
        """List all available configurations"""
        print("\n" + "="*60)
        print("Available Configurations")
        print("="*60)
        for key, config in self.configs.items():
            print(f"\n📋 {key.upper()}")
            print(f"   Name: {config['name']}")
            print(f"   Description: {config['description']}")
            print(f"   Gestures:")
            for gesture in config['gestures']:
                print(f"      • {gesture['label']}: {gesture['file']}")
        print("\n" + "="*60)
    
    def run_default(self):
        """Run with default configuration"""
        print("🚀 Running with default configuration...")
        subprocess.run(['python', 'umap_test.py'], check=True)
    
    def run_config(self, config_key):
        """Run with a specific configuration from file"""
        if config_key not in self.configs:
            print(f"Error: Configuration '{config_key}' not found")
            print("Available configurations:")
            for key in self.configs.keys():
                print(f"  - {key}")
            sys.exit(1)
        
        config = self.configs[config_key]
        print(f"\n🚀 Running with configuration: {config_key}")
        print(f"   Name: {config['name']}")
        print(f"   Description: {config['description']}")
        
        cmd = ['python', 'umap_test.py', '--config', self.config_file, '--config-key', config_key]
        subprocess.run(cmd, check=True)
    
    def run_custom(self, files, labels):
        """Run with custom files and labels"""
        if len(files) != len(labels):
            print("Error: Number of files must match number of labels")
            sys.exit(1)

        resolved_files = []
        for file_path in files:
            candidate = Path(file_path)
            if candidate.exists():
                resolved_files.append(str(candidate))
                continue

            csv_candidate = Path('CSV-Files') / file_path
            if csv_candidate.exists():
                resolved_files.append(str(csv_candidate))
                continue

            resolved_files.append(file_path)
        
        print("\n🚀 Running with custom configuration")
        print(f"   Files: {resolved_files}")
        print(f"   Labels: {labels}")
        
        cmd = ['python', 'umap_test.py', '--files'] + resolved_files + ['--labels'] + labels
        subprocess.run(cmd, check=True)
    
    def show_presets(self):
        """Show all preset commands"""
        print("\n" + "="*60)
        print("Quick Run Presets")
        print("="*60)
        presets = {
            'default': 'Default (Open/Close Hand)',
            'ricardo': 'Ricardo only data',
            'all': 'All subjects (Ricardo + Sirisha)',
            'emc': 'EMG_ASL folder data',
            'random': 'Random Forest dataset',
            'fingers': 'Individual finger tests',
            'hand': 'Hand position variants',
            'all_gestures': 'Complete gesture set',
            'comparison': 'Ricardo vs Mohak (Closed Hand)',
        }
        for cmd, desc in presets.items():
            print(f"  python run_umap.py {cmd:15} → {desc}")
        print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description='EMG UMAP Test Configuration Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_umap.py default
  python run_umap.py list
  python run_umap.py custom CSV-Files/Test-Ricardo.csv Ricardo EMG_ASL/CSV/Test-Ricardo_Open-Hand.csv "Open Hand"
        """
    )
    
    parser.add_argument(
        'action',
        nargs='?',
        default='default',
        help='Action to perform: default, list, presets, custom, or preset name (ricardo, all, emc, random)'
    )
    
    parser.add_argument(
        '--files',
        nargs='+',
        help='File paths for custom configuration (space-separated)'
    )
    
    parser.add_argument(
        '--labels',
        nargs='+',
        help='Labels for custom configuration (space-separated, must match number of files)'
    )
    
    args = parser.parse_args()
    
    runner = UMAPRunner()
    
    if args.action == 'default':
        runner.run_default()
    elif args.action == 'list':
        runner.list_configs()
    elif args.action == 'presets':
        runner.show_presets()
    elif args.action == 'custom':
        if not args.files or not args.labels:
            print("Error: --files and --labels required for custom configuration")
            sys.exit(1)
        runner.run_custom(args.files, args.labels)
    elif args.action in runner.configs:
        runner.run_config(args.action)
    elif args.action in ['ricardo', 'all', 'emc', 'random', 'fingers', 'hand', 'all_gestures', 'comparison']:
        # Handle preset shortcuts
        preset_mapping = {
            'ricardo': ['CSV-Files/Test-Ricardo.csv'], 
            'all': ['CSV-Files/Test-Ricardo.csv', 'CSV-Files/Test-Sirisha.csv'],
            'emc': ['EMG_ASL/CSV/Test-Ricardo_Open-Hand.csv', 'EMG_ASL/CSV/Test-Ricardo_Closed-Hand.csv'],
            'random': [
                'RandomForest/DATASET EMG MINDROVE/AYU/subjek_AYU1.csv',
                'RandomForest/DATASET EMG MINDROVE/DANIEL/subjek_DANIEL1.csv',
                'RandomForest/DATASET EMG MINDROVE/LINTANG/subjek_Lintang1.csv'
            ],
            'fingers': [
                'CSV-Files/index_finger.csv',
                'CSV-Files/ring_finger.csv',
                'CSV-Files/pinky_finger.csv'
            ],
            'hand': [
                'CSV-Files/closed_hand.csv',
                'CSV-Files/index_finger.csv',
                'CSV-Files/ring_finger.csv',
                'CSV-Files/pinky_finger.csv'
            ],
            'all_gestures': [
                'CSV-Files/Test-Ricardo_Open-Hand.csv',
                'CSV-Files/closed_hand.csv',
                'CSV-Files/index_finger.csv',
                'CSV-Files/ring_finger.csv',
                'CSV-Files/pinky_finger.csv'
            ],
            'comparison': [
                'CSV-Files/Test-Ricardo_Closed-Hand.csv',
                'CSV-Files/Test-Mohak_Closed-Hand.csv'
            ]
        }
        
        label_mapping = {
            'ricardo': ['Ricardo Mixed'],
            'all': ['Ricardo', 'Sirisha'],
            'emc': ['Open Hand', 'Close Hand'],
            'random': ['AYU-1', 'DANIEL-1', 'LINTANG-1'],
            'fingers': ['Index Finger', 'Ring Finger', 'Pinky Finger'],
            'hand': ['Closed Hand', 'Index Finger', 'Ring Finger', 'Pinky Finger'],
            'all_gestures': ['Open Hand', 'Closed Hand', 'Index Finger', 'Ring Finger', 'Pinky Finger'],
            'comparison': ['Ricardo - Closed Hand', 'Mohak - Closed Hand']
        }
        
        files = preset_mapping.get(args.action, [])
        labels = label_mapping.get(args.action, [])
        runner.run_custom(files, labels)
    else:
        print(f"Error: Unknown action '{args.action}'")
        runner.show_presets()
        print("\nRun 'python run_umap.py list' to see all configurations")
        sys.exit(1)

if __name__ == '__main__':
    main()
