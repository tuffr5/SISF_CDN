import glob
import time
import random
import tqdm
import os

def pretty_us(v: float) -> str:
    v *= 10 ** 6
    return f'{v:.2f}'

s = time.time()
files = glob.glob('*')
dt = time.time() - s
print(f'Glob Time [us]:', pretty_us(dt))

files_filtered = [x for x in files if not os.path.isdir(x)]
random.shuffle(files_filtered)

times = {
    'open': [],
    'close':[],
    'read':[]
}

N = 1000
N = min(len(files_filtered), N)

for f in tqdm.tqdm(random.sample(files_filtered, N)):
    s = time.time()
    f = open(f, 'rb')
    dt = time.time() - s
    times['open'].append(dt)

    s = time.time() 
    ####
    dt = time.time() - s
    times['read'].append(dt)

    s = time.time()
    f.close()
    dt = time.time() - s
    times['close'].append(dt)

print("=" * 25, f'N={N}')
for key,vs in times.items():
    avg = sum(vs) / N
    std = (sum( (v - avg)**2.0 for v in vs ) / N)**0.5

    print(key + ' [us]:', pretty_us(avg), '±', pretty_us(std))