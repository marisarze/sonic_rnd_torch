
import os
import sys
import json
import random
import math
import numpy as np
import random
import time
import cv2
import gc
from tqdm import tqdm, trange
from sys import getsizeof
from collections import deque
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import queue

from multiprocessing import Queue, Process, Lock
from rnd_models import *
from train_util import *
from tensorflow.keras.optimizers import SGD, Adam
#tf.compat.v1.disable_eager_execution()
class SonicAgent():
    def __init__(self, episodes_step=10, max_env_steps=None, state_len=1, gamma=0.999, batch_size=100, workers=4, render=False):
        self.game_name = 'SonicTheHedgehog-Genesis'
        self.map_name = 'LabyrinthZone.Act2'
        self.scenario = 'contest'
        self.timedelay = 12
        self.batch_size = batch_size
        self.csteps = 36000 / self.timedelay
        self.num_workers = workers
        self.iw = 1.0
        self.ew = 0.0
        self.epochs = 5
        self.count = 5
        self.actions = self.get_actions()
        self.action_space = len(self.actions)
        self.state_len = state_len
        self.width = 120
        self.height = 84
        self.state_shape = (self.state_len, self.height, self.width,3)
        self.lam = 0.0
        self.crange = 0.005
        self.epsilon = 0.1
        self.gamma = gamma
        self.egamma = gamma 
        self.igamma = gamma
        self.ignored_steps = 1
        self.cutted_steps = max(self.ignored_steps, 15)
        self.render = render
        self.default = np.float32
        self.adv = np.float32
        self.maxstat = 3000
        self.epoch_size = 10
        self.nonusable_steps = 0#math.ceil(math.log(1e-4, self.igamma))
        self.choosed_length = 0 * self.csteps + self.nonusable_steps
        self.memory_size = 1.1 * self.csteps + self.nonusable_steps
        self.horizon = 15 * self.csteps
        self.map_length_limit = 60000

        

        self.reusable_memory = []
        self.base_keys = {'states': np.uint8,
                    'erewards': np.float64,
                    'actions': np.uint8,
                    'coords': np.int32}
        self.all_keys = {'states': np.uint8,
                        'erewards': np.float64,
                        'actions': np.uint8,
                        'coords': np.int32,
                        'irewards': np.float64,
                        'eadvantages': np.float64,
                        'iadvantages': np.float64,
                        'evariance': np.float64,
                        'ivariance': np.float64,
                        'policy': np.float64,
                        'entropy': np.float64,
                        'etargets': np.float64,
                        'itargets': np.float64,
                        'advantages': np.float64,
                        'crange': np.float64}

        self.maps_path = r'D:\sonic_models\maps.json'
        self.stats_path = r'D:\sonic_models\stats.json'

    def run_train(self):
        lock = Lock()
        data = Queue()
        game_ids = Queue()
        render_que = Queue()

        self.get_stats()
        self.create_maps()
        render_que.put(True)
        for i in range(self.stats['episodes_passed'], 1000000):
            game_ids.put(i)
            if self.render:
                self.render_que.put(True)

        self.run_workers(lock, data, render_que, game_ids)       
        self.create_models()
        
        while True:
            print('-------------------------------------------------------------------')
            self.reinitialize_buffer()
            self.reduce_memory()
            self.get_new_result(data)
            self.update_memory()

            result = self.reusable_memory[-1]
            self.update_stats(result)
            self.update_istats(result)
            self.update_astats(result)
            self.update_maps(result)

            self.train_ireward_model()
            self.create_buffer()
            self.normalize_advantages()
            self.buffer['crange'] = self.compute_crange(self.buffer)
            self.shuffle_buffer()
            if self.stats['initialized']>1:
                self.train_vmodel(self.buffer['itargets'], self.ivmodel)
            
            if self.stats['initialized']>3:
                self.train_policy()

            self.save_stats()
            self.save_models()
            self.conditionally_stop_workers()

    
    def train_policy(self):
        print('training policy model...')
        optimizer = SGD(lr=1e-6, momentum=0.999)
        self.policy_model.compile(optimizer=optimizer)
        K.set_value(self.policy_model.optimizer.lr, self.stats['policy_lr'])

        old_policies = self.buffer['policy']
        losses = []
        losses.append(evaluate([self.buffer['states'], self.buffer['advantages'], self.buffer['policy'], self.buffer['crange']],
                                                self.policy_model,
                                                self.policy_loss_fn, 
                                                batch_size = self.batch_size,
                                                verbose=True))
        ratio = [0]
        available = np.mean(self.buffer['crange'])
        print('evaluated loss:', losses[-1])
        print('available crange_ratio:', available/ self.crange)
        while ratio[-1]<1.0:
            train_by_epoch([self.buffer['states'], self.buffer['advantages'], self.buffer['policy'], self.buffer['crange']],
                                                self.policy_model,
                                                self.policy_loss_fn, 
                                                batch_size = self.batch_size)
            new_policies = self.policy_model.predict(self.buffer['states'])
            diffs = np.max(np.abs(new_policies - old_policies), axis=-1)
            ratio.append(np.mean(diffs / self.buffer['crange'][:,0]))
            print('ratio', ratio[-1])
            if ratio[-1]-ratio[-2]<0.01:
                K.set_value(self.policy_model.optimizer.lr, self.policy_model.optimizer.lr*1.2)
        if len(losses) > 30:
            self.stats['policy_lr'] *= 1.1
        
    
    def train_vmodel(self, targets, model):
        print('training {} model...'.format(model.name))
        #optimizer = Adam(lr=1e-6)
        optimizer = SGD(lr=1e-6, momentum=0.9)
        model.compile(optimizer=optimizer)
        K.set_value(model.optimizer.lr, math.sqrt(self.steps/self.batch_size) * self.stats['vmodel_lr'])

        losses = []
        old_values = model.predict(self.buffer['states'])
        #self.buffer['itargets'] = self.buffer['itargets'] - 0.95 * (self.buffer['itargets'] - old_values)
        losses.append(evaluate([self.buffer['states'], targets], 
                                    model,
                                    self.critic_loss_fn, 
                                    batch_size = self.batch_size))
        print('evaluated loss:', losses[-1])        
        count = 0
        ratio = 1         
        while ratio>0.95:  #count<20:                
            history = train_by_epoch([self.buffer['states'], targets], 
                                    model,
                                    self.critic_loss_fn, 
                                    batch_size = self.batch_size)
            losses.append(history[-1])
            ratio = losses[-1]/losses[0]
            print('ratio', ratio)
            count += 1
            if ratio>2:
                self.stats['vmodel_lr'] /= 2.0
                break
        if len(losses) > 50:
            self.stats['vmodel_lr'] *= 1.1


    def train_ireward_model(self):
        print('training ireward_model...')
        optimizer = Adam(lr=1e-6)
        #optimizer = SGD(lr=1e-6, momentum=0.0)
        self.ireward_model.compile(optimizer=optimizer, loss='mse')
        
        steps = len(self.reusable_memory[-1]['states'])
        randomize = np.arange(steps)
        np.random.shuffle(randomize)
        states = self.reusable_memory[-1]['states'][randomize]

        fast, slow, _ = self.ireward_model.predict(states)
        fast_diff = 5e-2 * fast / self.csteps * steps
        slow_diff = 5e-3 * slow / self.csteps * steps
        fast_target = fast - fast_diff
        slow_target = slow - slow_diff
        initial = [fast, slow]
        diff = [fast_diff, slow_diff]
        target = [fast_target, slow_target]

        for i in range(2):
            losses = [evaluate([states, target[i], initial[i]], 
                                    self.ireward_model, 
                                    self.ireward_loss_fn,
                                    batch_size = self.batch_size,
                                    verbose=True,
                                    phase = i)]
            ratio = 1
            K.set_value(self.ireward_model.optimizer.lr, math.sqrt(steps/self.batch_size) * self.stats['ireward_model_lr'][i])
            while ratio>0.2: 
                history = train_by_epoch([states, target[i], initial[i]], 
                                    self.ireward_model, 
                                    self.ireward_loss_fn,
                                    batch_size = self.batch_size,
                                    verbose=True,
                                    phase = i)
                losses.append(history[-1])
                #change = np.mean(preloss - self.ireward_model.predict(states)) / (steps * self.istep)
                ratio = losses[-1]/losses[0]
                print('ratio', '{:.6E}'.format(ratio), 'phase:', i)
                if ratio>10:
                    self.stats['ireward_model_lr'][i] /= 2.0
                    break
            if len(losses) > 30:
                self.stats['ireward_model_lr'][i] *= 1.1
        

    
    #@tf.function
    def ireward_loss_fn(self, inputs, model, phase):
        states = inputs[0]
        target = inputs[1]
        initial = inputs[2]
        *pred, _ = model(states)
        loss = (pred[phase] - target) ** 2/ (initial+1e-11) ** 2
        loss = tf.reduce_sum(loss)
        return loss

    #@tf.function
    def policy_loss_fn(self, inputs, model):
        states = inputs[0]
        rewards = inputs[1]
        old_policies = inputs[2]
        crange = inputs[3]
        policy = model(states)
        policy = tf.clip_by_value(policy, old_policies-crange, old_policies+crange)
        #beta= 1/2/crange
        base_loss = K.sum(-rewards * policy, axis=-1)
        #inertia_loss = beta *  K.sum(K.abs(rewards), axis=-1) * K.sum(K.pow(policy-old_policies, 2), axis=-1)/self.action_space
        loss = tf.reduce_sum(base_loss)  #+ inertia_loss)
        return loss

    #@tf.function
    def critic_loss_fn(self, inputs, model):
        states = inputs[0]
        target = inputs[1]
        value = model(states)
        loss = K.square(value - target)#/(K.square(y_true - sigma_input)+1e-6) 
        #loss = K.clip(loss0, loss0-epsilon*sigma_input, loss0+epsilon*sigma_input)
        loss = tf.reduce_sum(loss)
        return loss


    def get_entropy(self, policy_sequence):
        policy_array = np.array(policy_sequence)
        entropy = -np.sum(policy_array * np.log(policy_array), axis=-1) / np.log(self.action_space)
        return entropy
    

    def get_irewards(self, states):
        states = self.process_states(states)
        _, _, ratio = self.ireward_model.predict(states)
        
        if self.stats['initialized'] > 0:
            ratio -= self.stats['ratio_imean']
            ratio /= 10 * self.stats['ratio_istd'] * self.csteps #* self.csteps
        irewards = np.zeros(ratio.shape) 
        irewards[:-1] = ratio[1:]
        irewards[-1] = ratio[0]
        # irewards[:-1] = ratio[1:] - ratio[:-1]
        # irewards[-1] = ratio[0] - ratio[-1] #losses[0] - losses[-1] 
        #irewards *= 1-self.igamma
        #mratio = np.clip(ratio, 0, 100)
        #irewards = self.iw * (1-self.igamma) * ratio
        #irewards = np.clip(irewards, 0, 10)
        return irewards


    def update_memory(self):
        for result in self.reusable_memory:
            steps = len(result['states'])
            result['policy'] = self.policy_model.predict(result['states'])
            result['entropy'] = self.get_entropy(result['policy'])
            result['irewards'] = self.get_irewards(result['states'])
            result['eadvantages'] = np.zeros((steps, self.action_space))
            result['etargets'] = np.zeros((steps, self.action_space))
            result['evariance'] = np.zeros((steps, 1))
            # result['eadvantages'], result['etargets'], result['evariance'] = self.compute_advantages(result['states'],
            #                                                                                         result['erewards'],
            #                                                                                         result['actions'],
            #                                                                                         self.evmodel,
            #                                                                                         self.egamma,
            #                                                                                         self.lam,
            #                                                                                         episodic=True,
            #                                                                                         trace=False)
            
            result['iadvantages'], result['itargets'], result['ivariance'] = self.compute_advantages(result['states'],
                                                                                                    result['irewards'],
                                                                                                    result['actions'],
                                                                                                    self.ivmodel,
                                                                                                    self.igamma,
                                                                                                    self.lam,
                                                                                                    episodic=False,
                                                                                                    trace=False)
            result['advantages'] = self.iw * result['iadvantages'] + self.ew * result['eadvantages']
            #result['crange'] = self.compute_crange(result)


    def create_buffer(self):
        for result in self.reusable_memory:    
            for key in result.keys():
                self.buffer[key].extend(np.copy(result[key]))
        for key in self.buffer.keys():
            self.buffer[key] = np.array(self.buffer[key], dtype=self.all_keys[key])
        self.steps = len(self.buffer['states'])


    def normalize_advantages(self):
        for i in range(self.steps):
            action_ind = self.buffer['actions'][i]
            adv = self.buffer['advantages'][i,action_ind]
            norm_adv = (adv - self.stats['amean']) / self.stats['astd']
            self.buffer['advantages'][i, action_ind] = norm_adv
             

    def compute_crange(self, result):
        steps = len(result['states'])
        cranges = np.zeros((steps, self.action_space))
        for i in range(steps):
            action_ind = result['actions'][i]
            absadv = np.abs(result['advantages'][i,action_ind])
            cranges[i] += absadv
        cranges *= self.crange
        cranges = np.clip(cranges, self.crange, 1000)
        return cranges


    def shuffle_buffer(self):
        randomize = np.arange(self.steps)
        np.random.shuffle(randomize)
        for key in self.buffer.keys():
            self.buffer[key] = self.buffer[key][randomize]


    def create_maps(self, path=r'D:\sonic_models\maps.json'):
        try:
            with open(path, 'r') as file:
                self.maps = json.load(file)
        except:
            keys = ('x', 'y', 'some_map', 'ireward_map', 'entropy_map', 'steps')
            self.maps = dict()
            for key in keys:
                self.maps[key] = []
    
    
    def update_maps(self, result, path=r'D:\sonic_models\maps.json'):
        steps = len(result['states'])
        self.maps['x'].extend(list(result['coords'][:-self.cutted_steps,0]))
        self.maps['y'].extend(list(result['coords'][:-self.cutted_steps,1]))
        self.maps['some_map'].extend(list(result['irewards'][:-self.cutted_steps]))
        self.maps['ireward_map'].extend(list(result['irewards'][:-self.cutted_steps]))
        self.maps['entropy_map'].extend(list(result['entropy'][:-self.cutted_steps]))
        self.maps['steps'].append(steps-self.cutted_steps)
        if len(self.maps['x']) > self.map_length_limit:
            for key in self.maps.keys():
                if key is not 'steps':
                    ind = int(self.maps['steps'][0])
                    self.maps[key] = self.maps[key][ind:]
            self.maps['steps'] = self.maps['steps'][1:] 
        self.transform2pythonic(self.maps)
        try:
            with open(path, 'w') as file:
                json.dump(self.maps, file, indent=4)
        except:
            print('Error occured for saving maps') 


    def update_istats(self, result):
        states = self.reusable_memory[-1]['states']
        fast, slow, ratio = self.ireward_model.predict(states)
        self.evaluate_stat_params(fast, 'fast_imean', 'fast_istd', 'fast_imax', 'fast_imin')
        self.evaluate_stat_params(slow, 'slow_imean', 'slow_istd', 'slow_imax', 'slow_imin')
        self.evaluate_stat_params(ratio, 'ratio_imean', 'ratio_istd', 'ratio_imax', 'ratio_imin')
        self.stats['initialized'] += 1


    def update_astats(self, result):
        advantages = result['advantages']
        adv = np.sum(advantages, axis=-1)
        self.evaluate_stat_params(adv, 'amean', 'astd', 'amax', 'amin')


    def evaluate_stat_params(self, sequence, average, deviation, maximum, minimum):
        mean = np.mean(sequence)
        std = np.std(sequence)
        maxi = np.max(sequence)
        mini = np.min(sequence)
        steps = len(sequence)
        all_steps = np.sum(self.stats['steps_list'])
        horizon = min(all_steps, self.horizon)
        self.stats[average] = ((horizon-steps)*self.stats[average]+steps*mean)/horizon
        self.stats[deviation] = ((horizon-steps)*self.stats[deviation]+steps*std)/horizon
        self.stats[maximum] = ((horizon-steps)*self.stats[maximum]+steps*maxi)/horizon
        self.stats[minimum] = ((horizon-steps)*self.stats[minimum]+steps*mini)/horizon


    def compute_advantages(self, states, rewards, actions, value_model, gamma, lam, episodic, trace):
        steps = len(states)
        states = self.process_states(states)
        values = value_model.predict(states)
        rewards = rewards.reshape((-1,1))

        advantages = np.zeros((steps,self.action_space)).astype(self.adv)
        target_values = np.zeros((steps,1)).astype(self.adv)
        td = np.zeros((steps,1)).astype(self.adv)

        if episodic:
            values = np.vstack((values[:], np.array([[0]])))
            target_values[:,0] = rewards[:,0] + gamma * values[1:,0]
            td[:,0] =  rewards[:,0] + gamma * values[1:,0] - values[:-1,0]
        else:
            for ind in range(steps):
                target_values[ind,0] = (values[ind-1,0] - rewards[ind-1,0])/gamma
            values = np.vstack((values[:], np.array([values[0,0]])))
            td[:,0] =  rewards[:,0] + gamma * values[1:,0] - values[:-1,0]
        # G = []
        # temp = 0
        # for reward in rewards[::-1]:
        #     temp = reward + gamma * temp
        #     G.append(temp)
        # G.reverse()
        # G = np.array(G, dtype=np.float32)
        
        
        
        # if self.stats['initialized']<1000000:
        # target_values[:,0] = G[:]
        
        
        temp = 0
        gl = gamma * lam
        ind = steps-1
        for elem in td[::-1,0]:
            temp = elem + gl * temp
            advantages[ind, actions[ind]] = temp        
            ind -= 1
        return advantages, target_values, td ** 2


    def run_episode(self, Amodel, render=False, record=False, game_id=0, path='.'):
        import retro
        import random
        if record:
            env = retro.make(game=self.game_name, state=self.map_name, scenario=self.scenario, record=path)    
        else:
            env = retro.make(game=self.game_name, state=self.map_name, scenario=self.scenario)
        env.movie_id = game_id
        cur_mem = deque(maxlen = self.state_len)
        done = False
        delay_limit = self.timedelay
        frame_reward = 0
        states = []
        erewards = []
        actions = []
        policy = []
        coords = []
        for _ in range(self.state_len):
            cur_mem.append(np.zeros((self.height, self.width, 3),dtype=self.default))
        next_state = env.reset()
        while not done:
            next_state = self.resize_state(next_state)
            cur_mem.append(next_state)
            states.append(np.array(cur_mem, dtype=np.uint8))
            action_id, cur_policy = self.choose_action(cur_mem, Amodel, render)
            policy.append(cur_policy)
            actions.append(action_id)
            for _ in range(random.randint(delay_limit-0, delay_limit+0)):
                next_state, reward, done, info = env.step(self.actions[action_id])
                frame_reward += reward
            erewards.append(frame_reward)
            coords.append([info['x'], info['y']])
            frame_reward = 0
            if render:
                env.render()
        env.render(mode='rgb_array', close=True)
        return {'states':states,
                'erewards':erewards,
                'actions': actions,
                'policy': policy,
                'coords': coords}
            

    def worker(self, l, data, render_que, game_ids, use_gpu=False):
        import os
        import time

        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        l.acquire()
        amodel = policy_net(self.state_shape, self.action_space)
        if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
            amodel.load_weights(r'D:\sonic_models\policy_model.h5')
            policy_loading_time = os.path.getmtime(r'D:\sonic_models\policy_model.h5')
        else:
            policy_loading_time=0
        l.release()

        while True:
            l.acquire()
            if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
                if policy_loading_time<os.path.getmtime(r'D:\sonic_models\policy_model.h5'):
                    policy_loading_time = os.path.getmtime(r'D:\sonic_models\policy_model.h5')
                    try:
                        amodel.load_weights(r'D:\sonic_models\policy_model.h5')
                        print('new weights are loaded by process', os.getpid())
                    except:
                        print('new weights are not loaded by process', os.getpid())
            l.release()
            game_id = game_ids.get()
            render = False
            if render_que.qsize()>0:
                render = render_que.get()
            # if game_id in self.special_game_ids:
            #     record_replay=True
            # else:
            record_replay=False
            result = self.run_episode(amodel, render=render, record=record_replay, game_id=game_id, path=r'D:\sonic_models\replays')
            while True:
                if data.qsize()<1:
                    data.put(result)
                    break
                else:
                    time.sleep(10)


    def choose_action(self, state_col, Amodel, render):
        states = np.reshape(np.array(state_col, dtype=np.uint8), (1, *self.state_shape))
        policy = Amodel.predict(states)
        policy = policy[-1]
        order = np.random.choice(self.action_space, size=None, p=policy)
        return order, policy


    def process_states(self, states):
        states = np.array(states, dtype=np.uint8)
        states.reshape((len(states), *self.state_shape))
        return states.astype(np.uint8)


    def get_actions(self):
        target = []
        buttons = ('B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z')
        #actions = [[], ['LEFT'], ['RIGHT'], ['LEFT','DOWN'], ['RIGHT','DOWN'], ['DOWN'], ['DOWN', 'B'], ['B'], ['LEFT','B'], ['RIGHT','B']]
        actions = [[], ['LEFT'], ['RIGHT'], ['LEFT','B'], ['RIGHT','B'], ['B'], ['DOWN']]
        for i in range(len(actions)):
            action_list = [0] * len(buttons)
            for button in actions[i]:
                action_list[buttons.index(button)]=1
            target.append(np.array(action_list))
        return np.array(target)    
        

    def resize_state(self, state):
        resized = cv2.resize(state, (self.width, self.height))
        resized = np.array(resized, dtype=np.uint8)
        return resized


    def process_ereward(self, ereward_memory):
        sume = np.sum(ereward_memory)
        #ereward_memory_temp = np.zeros((len(ereward_memory),))
        ereward_memory = np.array(ereward_memory)

        #for i in range(len(ereward_memory)-1):
        #    if np.sum(ereward_memory[:i])>0.02:
        #        ereward_memory_temp[i] = np.sum(ereward_memory[:i])
        #        ereward_memory[:i] *= 0
        #ereward_memory[:] *= 0
        #ereward_memory[:] = ereward_memory_temp[:]
        if sume>=0.99:
            ereward_memory[-1]=1
        else:
            pass #ereward_memory[-1]=10 #-sume/50
        return ereward_memory


    def get_stats(self, path=r'D:\sonic_models\stats.json'):
        try:
            with open(path, 'r') as file:
                self.stats = json.load(file)
        except:
            print('loading default stats...')
            self.stats = dict()
            self.stats['steps_list'] = []
            self.stats['episodes_passed'] = 0
            self.stats['initialized'] = 0

            self.stats['fast_imean'] = 0
            self.stats['fast_istd'] = 1
            self.stats['fast_imax'] = 1
            self.stats['fast_imin'] = 0
            
            self.stats['slow_imean'] = 0
            self.stats['slow_istd'] = 1
            self.stats['slow_imax'] = 1
            self.stats['slow_imin'] = 0

            self.stats['ratio_imean'] = 0
            self.stats['ratio_istd'] = 1
            self.stats['ratio_imax'] = 1
            self.stats['ratio_imin'] = 0

            self.stats['amean'] = 0
            self.stats['astd'] = 1
            self.stats['amax'] = 1
            self.stats['amin'] = 0
            self.stats['window'] = 10
            
            self.stats['mean100_external_rewards'] = []
            self.stats['external_rewards'] = []
            self.stats['entropy'] = []
            self.stats['action_std'] = []

            self.stats['policy_lr'] = 2e-5
            self.stats['vmodel_lr'] = 1e-2
            self.stats['ireward_model_lr'] = [5e-8, 5e-8]



    def save_stats(self, path=r'D:\sonic_models\stats.json'):
        self.transform2pythonic(self.stats)
        try:
            with open(path, 'w') as file:
                json.dump(self.stats, file, indent=4)
            print('Stats saved')
        except:
            print('Error occured for saving stats')
    

    def update_stats(self, result):
        steps = len(result['entropy'])
        self.stats['action_std'].append(np.std(result['policy']))
        self.stats['episodes_passed'] += 1
        self.stats['steps_list'].append(steps) 
        self.stats['entropy'].append(np.mean(result['entropy']))
        self.stats['external_rewards'].append(np.sum(result['erewards']))
        self.stats['mean100_external_rewards'].append(np.mean(self.stats['external_rewards'][-100:]))
        
        print('episode: ', self.stats['episodes_passed'],
            'episode_reward: ', self.stats['external_rewards'][-1],
            'mean_reward: ', self.stats['mean100_external_rewards'][-1])
        print('mean_entropy:', np.mean(self.stats['entropy'][-10:]))
        print('sum_irewards:', np.sum(result['erewards']))
        print('steps:', steps)
        print('itargets:', np.mean(result['itargets']))
        print('advantages_mean:', np.mean(np.sum(result['iadvantages'], axis=-1)))
    

    def reinitialize_buffer(self):
        self.buffer=dict()
        for key in self.all_keys:
            self.buffer[key]=list()    
    

    def get_memory_length(self):
        memory_length = 0
        for result in self.reusable_memory:
            memory_length += len(result['states'])
        return memory_length


    def reduce_memory(self):
        # while self.get_memory_length()>self.memory_size and len(self.reusable_memory)>2:
        #     # index = np.argmin(self.mean_adv)
        #     # self.reusable_memory.pop(index)
        #     # self.mean_adv.pop(index)
        #     self.reusable_memory.pop(0)
        while len(self.reusable_memory)>1:
            self.reusable_memory.pop(0)
        gc.collect()


    def save_models(self):
        self.policy_model.save(r'D:\sonic_models\policy_model.h5')
        #self.evmodel.save(r'D:\sonic_models\evalue_model.h5')
        self.ivmodel.save(r'D:\sonic_models\ivalue_model.h5')
        self.ireward_model.save(r'D:\sonic_models\ireward_model.h5')
    

    def create_models(self):
        #optimizer = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        #optimizer = SGD(lr=1e-6, momentum=0.99)
        if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
            self.policy_model = load_model(r'D:\sonic_models\policy_model.h5')
        else:
            self.policy_model = policy_net(self.state_shape, self.action_space)
            self.policy_model._name = 'policy_model'
        
        # if os.path.isfile(r'D:\sonic_models\evalue_model.h5'):
        #     self.evmodel = load_model(r'D:\sonic_models\evalue_model.h5')
        # else:
        #     self.evmodel = critic_net(self.state_shape, self.epsilon)
        #     self.evmodel._name = 'evmodel'
        #     optimizer = SGD(lr=1e-6, momentum=0.9)
        #     self.evmodel.compile(optimizer=optimizer)
        #     
        #     weights = [np.array(w) for w in self.evmodel.get_weights()]
        #     weights[-1] *= 0
        #     self.evmodel.set_weights(weights)            

        if os.path.isfile(r'D:\sonic_models\ivalue_model.h5'):
            self.ivmodel = load_model(r'D:\sonic_models\ivalue_model.h5')
        else:
            self.ivmodel = critic_net(self.state_shape, self.epsilon)
            self.ivmodel._name = 'ivmodel'
        
        if os.path.isfile(r'D:\sonic_models\ireward_model.h5'):
            self.ireward_model = load_model(r'D:\sonic_models\ireward_model.h5')
        else:
            self.ireward_model = reward_net(self.state_shape)
            self.ireward_model._name = 'ireward_model'

    def get_new_result(self, data):
        result = data.get()
        for key in self.base_keys.keys():
            result[key] = np.array(result[key], dtype=self.base_keys[key])
        self.reusable_memory.append(result)


    def run_workers(self, lock, data, render_que, game_ids):
        processes = []
        for i in range(self.num_workers):
            processes.append(Process(target=self.worker, args=(lock, data, render_que, game_ids)))
        
        for p in processes:
            p.start()
    

    def conditionally_stop_workers(self):
        if self.stats['mean100_external_rewards'][-1]>99500:
            print('agent completed training successfully in '+str(self.stats['episodes_passed'])+' episodes')
            for i in range(self.num_workers):
                last_word = 'process '+str(self.processes[i].pid)+' terminated'
                self.processes[i].terminate()
                print(last_word)
               
    def transform2pythonic(self, dictionary):
        for key in dictionary.keys():
            if isinstance(dictionary[key], list):
                dictionary[key] = list(map(lambda x: float(x), dictionary[key]))
            elif isinstance(dictionary[key], np.generic):
                dictionary[key] = float(dictionary[key])


if __name__ == '__main__':
    agent = SonicAgent()
    agent.run_train()
    # result_file = open(r'D:\sonic_models\result.json', 'r')
    # result = json.load(result_file)
    # episode_passed = result['episode_passed']
    # Amodel = create_policy_net('policy', 1)
    # if os.path.isfile(r'D:\sonic_models\policy_model.h5'):
    #     Amodel.load_weights(r'D:\sonic_models\policy_model.h5')
    # agent.run_episode(Amodel, render=True, record=True, game_id=episode_passed, path=r'D:\sonic_models\replays')


# def train_policy(self):
#         print('training policy model...')
#         temp_adv = np.sum(self.buffer['advantages'], axis=-1).reshape((-1,))
#         bool_adv = temp_adv>0
#         pos_len = len(bool_adv[bool_adv])
#         print('pos_parts', pos_len/ len(temp_adv))
#         if pos_len>0:
#             losses = []
#             ef = min(len(self.buffer['states'][bool_adv])/self.memory_size, 1)
#             mask = []
#             for i, adv in enumerate(self.buffer['advantages'][bool_adv]):
#                 elem = adv/np.sum(adv, axis=-1)
#                 mask.append(elem)
#                 #self.buffer['advantages'][bool_adv][i] = elem 
#             mask = np.array(mask)
#             masked_policy = (self.buffer['policy'][bool_adv]) * mask
#             choosed_policy = np.sum(masked_policy, axis=-1)
#             ratios = (1-choosed_policy)/(1-1/self.action_space)
#             ratios = np.clip(ratios, 0, 1)
#             crange = np.zeros((len(self.buffer['states'][bool_adv]),1))
#             crange[:,0] += self.crange * ratios * ef
#             count = 0
#             steps = len(self.buffer['states'][bool_adv])
#             K.set_value(self.policy_model.optimizer.lr, self.stats['policy_lr'] * math.sqrt(steps))
#             old_policies = self.buffer['policy'][bool_adv]
#             losses.append(self.policy_loss_fn([self.buffer['states'][bool_adv], self.buffer['advantages'][bool_adv], old_policies, crange],
#                                                     self.policy_model,
#                                                     evaluate=True)/steps)
#             print('evaluated loss:', losses[-1])
#             while count<20:
#                 randomize = np.arange(len(self.buffer['advantages'][bool_adv]))
#                 np.random.shuffle(randomize)
#                 history = train_by_epoch([self.buffer['states'][bool_adv], self.buffer['advantages'][bool_adv], old_policies, crange],
#                                                     self.policy_model,
#                                                     self.policy_loss_fn, 
#                                                     batch_size = self.batch_size, epochs=1)
#                 losses.append(history[-1])
#                 peak = np.argmax(losses)
#                 count += 1
#                 if len(losses)> peak+self.count+1:
#                     diffl = np.array(losses[:-1]) - np.array(losses[1:])
#                     ratio = np.mean(diffl[-self.count:])/np.mean(diffl[peak:peak+self.count])
#                     # if diffl[-1]/diffl[-2] > 0.5:
#                     #     self.stats['policy_lr'] = np.clip(1.2 * self.stats['policy_lr'], 1e-5, 1e-1)
#                     # else:
#                     #     self.stats['policy_lr'] = np.clip(0.8 * self.stats['policy_lr'], 1e-5, 1e-1)
#                     # K.set_value(self.policy_model.optimizer.lr, self.stats['policy_lr'])
#                     print('ratio:', round(ratio, 6), 'lr', round(self.stats['policy_lr'], 6))
#                     if ratio<0.3 and losses[-1]<losses[0]:
#                         break


            # fullness = self.get_memory_length() / self.memory_size
            # print('memory_fullness', fullness, 'memory_len', len(self.reusable_memory), 'size', getsizeof(self.reusable_memory)/1024**2)
            


                # def custom_loss(y_pred, reward_input, reference, old_reference, crange):
    #     xmin = 0.99
    #     beta= 1/2/crange
    #     #entropy = -K.sum(y_pred * K.log(y_pred), axis=-1) / K.log(tf.constant(action_space, tf.float32))
    #     # #ratio = (y_pred+1e-2)/(old_input+1e-2)
    #     base_loss = K.sum(-reward_input * y_pred, axis=-1)
    #     inertia_loss = beta *  K.sum(K.abs(reward_input), axis=-1) * K.sum(K.pow(reference-old_reference, 2), axis=-1)/tf.constant(action_space, tf.float32)
    #     #entropy_loss = 0.06 * beta * K.sum(K.abs(reward_input), axis=-1) * K.sum(y_pred ** 4, axis=-1)/tf.constant(action_space, tf.float32)
    #     # loss0 = K.sum(-reward_input * old_input + beta * K.abs(reward_input) * K.pow(old_input, 2), axis=-1)
    #     # ymin = K.sum(-reward_input * xmin + beta * K.abs(reward_input) * xmin ** 2, axis=-1)
    #     # minl = K.maximum(ymin, loss0+dnl)
    #     # maxl = loss0 + dpl
    #     # loss = K.clip(loss, minl, maxl)
    #     # entropy_old = -K.sum(old_input * K.log(old_input), axis=-1) / K.log(tf.constant(action_space, tf.float32))
    #     # d = K.pow(entropy_old, 1) * crange
    #     # d = tf.reshape(d, [tf.shape(d)[0],1])
    #     # d = crange
    #     # pg_loss2 =-reward_input * K.clip(ratio, 1-d, 1+d)
    #     # pg_loss = K.maximum(pg_loss1,pg_loss2)
    #     loss = base_loss + inertia_loss #+ entropy_loss 
    #     return loss