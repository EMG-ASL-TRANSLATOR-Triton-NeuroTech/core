"""data_logger.py
Reusable DataLogger class for saving labeled multi-channel EMG data rows.
Each row format: timestamp, chan_1, chan_2, ..., chan_N, label
"""
import csv, os
from datetime import datetime

class DataLogger:
    def __init__(self, out_dir='data', session_name=None, channels=4):
        os.makedirs(out_dir, exist_ok=True)
        if session_name is None:
            session_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.filename = os.path.join(out_dir, f'{session_name}.csv')
        self.channels = channels
        # write header
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['timestamp'] + [f'chan_{i+1}' for i in range(channels)] + ['label']
            writer.writerow(header)

    def write_row(self, timestamp, channels_values, label=''):
        # ensure correct length
        row = [timestamp] + [float(ch) for ch in channels_values[:self.channels]]
        row += [''] * max(0, self.channels - len(channels_values))
        row.append(label)
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

if __name__ == '__main__':
    # small demo
    dl = DataLogger(out_dir='data', session_name='demo_session', channels=4)
    dl.write_row(0.0, [0.1,0.2,0.3,0.4], 'fist')
