import imageio
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num', default=100, type=int)
parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--fps', default=24, type=int)
options = parser.parse_args()

from img_resize import test2

test2()

# gif = []
# path = f"samples"

# for filename in tqdm(range(options.num)):
#         gif.append(imageio.imread(f"{path}/sample_epoch_{filename}.jpg"))

# imageio.mimsave(f"gifs/{options.num}_{options.batch_size}.gif", gif, fps=options.fps)