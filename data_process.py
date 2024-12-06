import numpy as np
import os
import cv2

DATA_DIR = 'data/iPad_scans/2024_08_08_10_11_23'
new_DATA_DIR = 'data/iPad_processed/2024_08_08_10_11_23'

os.makedirs(new_DATA_DIR, exist_ok=True)

temp_filenames = os.listdir(DATA_DIR)
print('temp_filenames:', temp_filenames)
frame_filenames = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.jpg') and 'frame' in f]
frame_filenames.sort()

data_num = len(frame_filenames)
print('data_num:', data_num)

selected_index = np.arange(0, data_num, 1)

for i in selected_index:
    print('Processing', frame_filenames[i])
    frame = cv2.imread(frame_filenames[i])
    cv2.imwrite(os.path.join(new_DATA_DIR, 'frame_%06d.png' % i), frame)
    print('Saved to', os.path.join(new_DATA_DIR, 'frame_%06d.png' % i))

