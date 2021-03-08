from tqdm import tqdm
import time
pbar = tqdm(["a", "b", "c", "d"], ncols=100)
count = 0
for char in pbar:
    count += count ** 2
    pbar.set_description("Processing %s" % char)
    pbar.set_postfix({'count square': count})
    time.sleep(0.1)