import imageio
from tqdm import tqdm

gif = []
path = f"samples"

for filename in tqdm(range(2100)):
        gif.append(imageio.imread(f"{path}/sample_epoch_{filename}.jpg"))

imageio.mimsave(f"gifs/2100.gif", gif, fps=120)