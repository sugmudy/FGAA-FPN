import os
import cv2
import numpy as np
img_dir = r'D:\mmrotate-main\data\split_DOTA\train\images'
label_dir = r'D:\mmrotate-main\data\split_DOTA\train\labelTxt'
out_dir = r'D:\mmrotate-main\data\split_DOTA\vis_check'

os.makedirs(out_dir, exist_ok=True)

def draw_poly(img, coords, color=(0, 255, 0)):
    pts = coords.reshape((-1, 1, 2)).astype(int)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

for name in os.listdir(img_dir)[:10]:  # 只显示前10张
    if not name.endswith('.png'):
        continue
    img_path = os.path.join(img_dir, name)
    label_path = os.path.join(label_dir, name.replace('.png', '.txt'))

    img = cv2.imread(img_path)
    if img is None:
        continue

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                if line.startswith('i') or line.startswith('g'):
                    continue
                parts = line.strip().split()
                coords = [float(x) for x in parts[:8]]
                coords = np.array(coords).reshape(4, 2)
                draw_poly(img, coords, color=(0, 255, 0))

    cv2.imwrite(os.path.join(out_dir, name), img)
