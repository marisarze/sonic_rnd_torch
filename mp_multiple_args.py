
import numpy as np
from tqdm import trange
import time
from multiprocessing import Queue, Process, Lock, Pool

def some_func(rewards, values, dummy, dummy2, dummy3):
    return np.array([reward+0.999 * value for reward, value in zip(rewards, values)])

if __name__ == '__main__':
    
    # target_values = np.random.rand(3001, 10)
    # actions = np.ones((3000,), dtype=np.uint8)
    # rewards = np.random.rand(3000)
    # actions = np.hstack((actions[:], actions[0]))
    # steps = len(rewards)
    # gamma = 0.999

    target_values = np.random.rand(30001,)
    rewards = np.random.rand(30000)
    steps = len(rewards)
    gamma = 0.999    

    num_processes = 8
    p = Pool(num_processes)
    start = time.time()
    for ind in trange(3000, ncols=130, disable=False):
        target_values2 = np.copy(target_values[1:])
        coeff = int(steps/num_processes)+1
        tlist2 = [np.copy(target_values2[ind*coeff:(ind+1)*coeff]) for ind in range(num_processes)]
        rlist = [np.copy(rewards[ind*coeff:(ind+1)*coeff]) for ind in range(num_processes)]
        dummy = [np.copy(rewards[ind*coeff:(ind+1)*coeff]) for ind in range(num_processes)]
        dummy2 = [np.copy(rewards[ind*coeff:(ind+1)*coeff]) for ind in range(num_processes)]
        dummy3 = [np.copy(rewards[ind*coeff:(ind+1)*coeff]) for ind in range(num_processes)]
        total = np.hstack(p.starmap(some_func, zip(rlist, tlist2, dummy, dummy2, dummy3)))
        target_values[:-1] = total
        target_values[-1] = target_values[0]
    print('compl:', time.time()-start)

    start = time.time()
    for ind in trange(3000, ncols=130):
        for i in range(steps):
            target_values[i] = rewards[i] + gamma * target_values[i+1]
        target_values[-1] = target_values[0]
    print('compl:', time.time()-start)

    # p = Pool(2)
    # s = p.starmap(f, zip(a,b))
    # print(s)
