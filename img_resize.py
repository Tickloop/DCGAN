import cv2
from tqdm import tqdm
import os

def main():
    for path, dirs, files in os.walk('data/cropped'):
        for file in tqdm(files):
            img = cv2.imread(f"{path}/{file}")
            img = cv2.resize(img, (128, 128))
            cv2.imwrite(f"data/downsized/{file}", img)

if __name__ == "__main__":
    main()