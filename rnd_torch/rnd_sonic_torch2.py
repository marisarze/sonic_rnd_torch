
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
import pickle
import torch
import PIL.ImageOps
from PIL import Image
from torch.optim import SGD, Adam
from tqdm import tqdm, trange
from sys import getsizeof
from collections import deque
from matplotlib import pyplot as plt
import queue
import torchvision
from multiprocessing import Queue, Process, Lock, Pool
from models2 import *
from training_utils import *

class SonicAgent():
    def __init__(self, episodes_step=10, max_env_steps=None,
                state_len=1, gamma=0.999, batch_size=50,
                workers=8, computers=12, render=False, use_gpu=True):
        self.game_name = 'SonicTheHedgehog-Genesis'
        self.map_name = 'LabyrinthZone.Act2'
        self.scenario = 'contest'
        self.timedelay = 12
        self.batch_size = batch_size
        self.csteps = int(36000 / self.timedelay)
        self.num_workers = workers
        self.num_computers = computers
        self.iw = 1.0
        self.ew = 0.0
        self.ereward_episodic = True
        self.ireward_episodic = False
        self.epochs = 5
        self.count = 5
        self.actions = self.get_actions()
        self.action_space = len(self.actions)
        self.state_len = state_len
        self.width = 120
        self.height = 84
        self.aug = torchvision.transforms.ColorJitter(brightness=0.5,
            contrast=0.5, saturation=0.5, hue=0.5)
        self.state_shape = (self.state_len, 3, self.height, self.width)
        self.lam = 0.0
        self.crange = 20/self.csteps
        self.epsilon = 0.1
        self.gamma = gamma
        self.egamma = gamma
        self.igamma = gamma
        self.multi = 4
        self.ignored_steps = 1
        self.cutted_steps = 13
        self.render = render
        self.advanced_type = torch.float32
        self.default = np.float32
        self.adv = np.float32
        self.maxstat = 3000
        self.epoch_size = 10
        self.memory_size = 20 * self.csteps
        self.horizon = 20 * self.csteps
        self.map_length_limit = 60000
        self.safe_episodes = 10
        self.use_gpu = use_gpu

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
                        'raw_iadvantages': np.float64,
                        'raw_eadvantages': np.float64,
                        'advantages': np.float64,
                        'crange': np.float64}

        self.lock = Lock()
        self.data = Queue()
        self.game_ids = Queue()
        self.render_que = Queue()
        self.input_que = Queue()
        self.result_que = Queue()

        self.maps_path = r'D:\sonic_models\maps.json'
        self.stats_path = r'D:\sonic_models\stats.json'

    def run_train(self):
        self.get_stats()
        self.render_que.put(True)
        for i in range(self.stats['episodes_passed'], 1000000):
            self.game_ids.put(i)
            if self.render:
                self.render_que.put(True)

        self.run_workers(self.lock, self.data, self.render_que, self.game_ids, use_gpu=self.use_gpu)
        #self.run_computers()
        self.create_models()
        self.load_memory()
        self.train_phase2()

    def train_phase2(self):
        self.phase = 0
        self.reinitialize_buffer()
        while True:
            print('-------------------------------------------------------------------')
            result = self.get_new_result(self.data)
            self.stats['episodes_passed'] += 1
            print('episode: ', self.stats['episodes_passed'],
            'episode_reward: ', np.sum(result['erewards']))
            self.complete_result(result)
            self.reusable_memory.append(result)

            steps_passed = self.get_memory_length() / self.memory_size
            print('steps_passed:', steps_passed)
            if steps_passed > 1:
                self.create_buffer(self.reusable_memory)
                self.update_stats()
                self.update_maps()
                self.shuffle_buffer()

                self.train_vmodel(self.buffer['itargets'], self.ivmodel)
                #self.train_vmodel(self.buffer['etargets'], self.evmodel)
                self.train_policy()
                self.train_ireward_model()

                self.reinitialize_buffer()
                self.reusable_memory = []


                self.save_models()

                self.save_stats()
                self.save_memory()
                self.render_que.put(False)
            # if self.stats['episodes_passed'] % 50 == 0:

            #     self.save_memory()

    def train_policy(self):
        print('training policy model...')
        amean = np.mean(np.sum(self.buffer['advantages'], axis=-1))
        astd = np.std(np.sum(self.buffer['advantages'], axis=-1))
        for ind, elem in enumerate(self.buffer['advantages']):
            self.buffer['advantages'][ind, self.buffer['actions'][ind]] -= amean
        self.buffer['advantages'] /= astd
        choosed = np.sum(self.buffer['advantages'], axis=-1) > -30
        # from scipy.stats.kde import gaussian_kde
        # kde = gaussian_kde(np.sum(self.buffer['advantages'], axis=-1))
        # lspace = np.linspace(-2.5,2.5,1000)
        # plt.plot(lspace, np.abs(lspace) * kde(lspace))
        # plt.show()

        
        # for ind, elem in enumerate(self.buffer['advantages']):
        #     temp = self.buffer['advantages'][ind, self.buffer['actions'][ind]]
        #     self.buffer['advantages'][ind, self.buffer['actions'][ind]] = np.sign(temp) * temp ** 3

        # amean = np.mean(np.sum(self.buffer['advantages'], axis=-1))
        # astd = np.std(np.sum(self.buffer['advantages'], axis=-1))
        # # for ind, elem in enumerate(self.buffer['advantages']):
        # #     self.buffer['advantages'][ind, self.buffer['actions'][ind]] -= amean
        # self.buffer['advantages'] /= astd

        # kde2 = gaussian_kde(np.sum(self.buffer['advantages'], axis=-1))
        # #fig = plt.figure()
        # lspace2 = np.sign(lspace) * lspace ** 3
        # plt.plot(lspace, np.abs(lspace) * kde2(lspace))
        # plt.show()

        old_policies = self.buffer['policy']
        ratio = [0]
        available = np.mean(self.buffer['crange']) #* self.buffer_steps / self.csteps
        #optimizer = Adam(self.policy_model.parameters(), lr=self.stats['policy_lr'])
        optimizer = SGD(self.policy_model.parameters(), lr=self.stats['policy_lr']) #* self.buffer_steps / self.csteps)
        print('available crange_ratio:', available/ self.crange)
        start = time.time()
        while len(ratio)<2: #ratio[-1] < 1.0:
            train_by_epoch([self.buffer['states'][choosed], self.buffer['advantages'][choosed], self.buffer['policy'][choosed], self.buffer['crange'][choosed]],
                                                self.policy_model,
                                                self.policy_loss_fn,
                                                optimizer=optimizer,
                                                batch_size=self.batch_size)
            new_policies = predict([self.buffer['states']], self.policy_model)
            diffs = np.max(np.abs(new_policies - old_policies), axis=-1)#/self.buffer['entropy']
            ratio.append(np.mean(diffs / available))
            print('ratio', ratio[-1])
            if time.time()-start>300 * len(self.buffer['states'])/ self.csteps:
                break
        if ratio[-1] < 1.0:
            self.stats['policy_lr'] *= 1.2
        else:
            self.stats['policy_lr'] /= 1.2

    def train_vmodel(self, targets, model):
        print('training {} model...'.format(model.name))
        batch_size = 50
        error_limit = 0.0
        method = train_by_batch
        #optimizer = Adam(model.parameters(), lr=1e-5)
        optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)

        losses = []
        old_values = predict([self.buffer['states']], model)
        mod_targets = self.buffer['itargets'] - error_limit * (self.buffer['itargets'] - old_values)
        losses.append(evaluate([self.buffer['states'], mod_targets],
                                model,
                                self.critic_loss_fn,
                                batch_size=self.batch_size))
        print('evaluated loss:', losses[-1])
        count = 0
        ratio = 1
        start = time.time()
        while len(losses)<2:#ratio > 0.9:  #count<20:
            history = method([self.buffer['states'], mod_targets],
                                    model,
                                    self.critic_loss_fn,
                                    optimizer=optimizer,
                                    batch_size=batch_size)
            losses.append(history[-1])
            ratio = losses[-1]/losses[0]
            print('ratio', ratio)
            count += 1
            if time.time()-start>120 * len(self.buffer['states'])/ self.csteps:
                break


    def train_ireward_model(self):
        print('training ireward_model...')
        optimizer = Adam(self.ireward_model.parameters(), lr=1e-4)
        #optimizer = SGD(self.ireward_model.parameters(), lr=1e-2, momentum=0.0) #* self.stats['fast_imean'], momentum=0.0)
        batch_size = 50
        steps = len(self.buffer['states'])
        states = self.buffer['states']
        initial = predict([states], self.ireward_model, convert_type=self.advanced_type)
        self.importance = 2.5
        coeffs = np.exp(self.importance * (initial-self.stats['fast_imean'])/self.stats['fast_istd'])
        self.cmean = np.mean(coeffs)
        start = time.time()
        losses = [1]
        ratio = 1
        #while ratio > 0.9999:
        train_by_batch([states, coeffs],
                    self.ireward_model,
                    self.ireward_loss_fn,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    verbose=True,
                    convert_type=self.advanced_type,
                    epochs=1)
        new_pred = predict([states], self.ireward_model, convert_type=self.advanced_type)
        nloss = np.mean(new_pred / initial)
        losses.append(nloss)
        #change = np.mean(preloss - self.ireward_model.predict(states)) / (steps * self.istep)
        ratio = losses[-1]/losses[0]
        print('ratio', '{:.6E}'.format(ratio))
        # if ratio>0.98:
        #     self.stats['ireward_model_lr'] *= 1.2
    #         break
            # if time.time()-start>120 * len(states)/ self.csteps:
            #     break




    def ireward_loss_fn(self, inputs, model):
        states = inputs[0]
        coeffs = inputs[1]
        mean = self.stats['fast_imean']
        std = self.stats['fast_istd']
        cmean = self.cmean
        pred = model(states)
        #loss = (pred - target) ** 2 / (initial+1e-11) ** 2
        #loss = pred * torch.exp(self.importance * (pred-mean)/std)/cmean
        loss = pred
        return loss

    # def ireward_loss_fn(self, inputs, model):
    #     states = inputs[0]
    #     #target = inputs[1]
    #     #initial = inputs[2]
    #     with torch.no_grad():
    #         initial = model(states)
    #     target = initial - self.stats['fast_istd']
    #     gpred = model(states)
    #     loss = (gpred - target) ** 1 #/ (initial+1e-11) ** 2
    #     return loss


    def policy_loss_fn(self, inputs, model, **kwargs):
        states = inputs[0]
        rewards = inputs[1]
        old_policies = inputs[2]
        crange = inputs[3]
        policy = model(states).double()
        #policy = torch.min(policy, old_policies+crange)
        #policy = torch.max(policy, old_policies-crange)
        #policy = torch.clip(policy, old_policies-crange, old_policies+crange)
        #beta= 1/2/crange
        base_loss = torch.sum(-rewards * policy, dim=-1)
        #inertia_loss = beta *  K.sum(K.abs(rewards), axis=-1) * K.sum(K.pow(policy-old_policies, 2), axis=-1)/self.action_space
        #loss = torch.sum(base_loss)  #+ inertia_loss)
        return base_loss


    def critic_loss_fn(self, inputs, model, **kwargs):
        states = inputs[0]
        targets = inputs[1]
        predicted = model(states)
        loss = torch.square(targets-predicted)
        loss = torch.mean(loss, dim=-1)
        return loss

    def get_entropy(self, policy_sequence):
        policy_array = np.array(policy_sequence)
        entropy = -np.sum(policy_array * np.log(policy_array), axis=-1) / np.log(self.action_space)
        return entropy


    # def get_irewards(self, states):
    #     states = self.process_states(states)
    #     fast = predict([states], self.ireward_model)
    #     fast /= 10 * self.stats['fast_istd']
    #     irewards = np.zeros_like(ratio)
    #     irewards[:-1] = ratio[1:] - ratio[:-1]
    #     irewards[-1] = ratio[0] - ratio[-1]
    #     return irewards

    def get_irewards(self, states):
        states = self.process_states(states)
        ratio = predict([states], self.ireward_model, convert_type=self.advanced_type)
        #ratio -= self.stats['fast_imean']
        #ratio /= 10 * self.stats['fast_istd'] * self.csteps
        self.stats['fast_imean'] = np.mean(ratio)
        self.stats['fast_istd'] = np.std(ratio)
        ratio -= self.stats['fast_imean']
        ratio /= self.stats['fast_istd']
        # perces = [np.percentile(ratio, x) for x in range(100)]
        # plt.plot(perces)
        # plt.show()
        xperc = np.percentile(ratio, 80)-0.5

        def new_func(x, x0, m):
            a = m/(x0+1)
            b = x0*(1-m/(x0+1))
            return a*x+b

        ratio = np.where(ratio>xperc, xperc+np.exp(-2*(ratio-xperc)) * (ratio-xperc), new_func(ratio, xperc, 0.183))
        runmean = np.mean(ratio[-50:])
        means = np.zeros_like(ratio)
        for i, elem in enumerate(ratio):
            means[i] = (elem + 49 * runmean)/50
            runmean = means[i]

        irewards = np.zeros_like(ratio)
        irewards[:-1] = means[1:] - means[:-1]#+ np.abs(ratio[1:]-ratio[:-1])
        irewards[-1] = means[0] - means[-1]#+ np.abs(ratio[0]-ratio[-1])

        #ratio = np.exp(-np.abs(ratio)) * ratio
        # ratio = np.where(ratio>0, 4*np.exp(-2*np.abs(ratio)) * np.tanh(ratio), np.tanh(ratio))
        # from scipy.stats.kde import gaussian_kde
        # kde = gaussian_kde(ratio)
        # lspace = np.linspace(-6.5,6.5,1000)
        # plt.plot(lspace, kde(lspace))
        # plt.show()

        

        # limit = 100.0
        # nratio = ratio[ratio<limit]
        # ratio = np.where(ratio>limit, np.mean(nratio), ratio)
        #ratio = 1-np.exp(-ratio)
        
        #ratio = np.clip(ratio, -5, 1.5)
        #ratio *= 1-self.gamma
        # irewards = np.zeros_like(ratio)
        # irewards[:-1] = ratio[1:] - ratio[:-1]#+ np.abs(ratio[1:]-ratio[:-1])
        # irewards[-1] = ratio[0] - ratio[-1]#+ np.abs(ratio[0]-ratio[-1])
        #irewards = np.where(irewards>0, np.exp(-2*np.abs(irewards)) * irewards, irewards)
        return irewards


    def update_memory(self, full=False):
        print('updating memory....')
        for result in tqdm(self.reusable_memory, ncols=130):
            self.complete_result(result)
        print('updating memory completed')


    def complete_result(self, result):
        steps = len(result['states'])
        result['policy'] = predict([result['states']], self.policy_model)
        result['entropy'] = self.get_entropy(result['policy'])
        #result['irewards'] = self.get_irewards(result['states'])
        # result['irewards'][-12:] = 0
        # result['raw_eadvantages'], result['etargets'], result['evariance'] = self.compute_advantages(result['states'],
        #                                                                                         result['erewards'],
        #                                                                                         result['actions'],
        #                                                                                         self.evmodel,
        #                                                                                         self.egamma,
        #                                                                                         self.lam,
        #                                                                                         episodic=self.ereward_episodic,
        #                                                                                         trace=False)

        # result['raw_iadvantages'], result['itargets'], result['ivariance'] = self.compute_advantages(result['states'],
        #                                                                                         result['irewards'],
        #                                                                                         result['actions'],
        #                                                                                         self.ivmodel,
        #                                                                                         self.igamma,
        #                                                                                         self.lam,
        #                                                                                         episodic=self.ireward_episodic,
        #                                                                                         trace=False)
        result['raw_eadvantages'] = np.zeros((steps, self.action_space))
        result['etargets'] = np.zeros((steps, self.action_space))#predict([result['states']], self.evmodel)
        result['evariance'] = np.zeros((steps, self.action_space))
        result['raw_iadvantages'] = np.zeros((steps, self.action_space))
        result['itargets'] = np.zeros((steps, 1)) #predict([result['states']], self.ivmodel)
        result['ivariance'] = np.zeros((steps, self.action_space))

        result['iadvantages'] = result['raw_iadvantages']
        result['eadvantages'] = result['raw_eadvantages']
        result['advantages'] = self.iw * result['iadvantages'] + self.ew * result['eadvantages']
        result['crange'] = self.compute_crange(result)

    def create_buffer(self, memory_seq):
        for result in memory_seq:
            for key in result.keys():
                self.buffer[key].extend(np.copy(result[key]))
        for key in self.buffer.keys():
            self.buffer[key] = np.array(self.buffer[key], dtype=self.all_keys[key])
        self.buffer['irewards'] = self.get_irewards(self.buffer['states'])
        self.buffer['iadvantages'], self.buffer['itargets'], self.buffer['ivariance'] = self.compute_advantages(self.buffer['states'],
                                                                                                self.buffer['irewards'],
                                                                                                self.buffer['actions'],
                                                                                                self.ivmodel,
                                                                                                self.igamma,
                                                                                                self.lam,
                                                                                                episodic=self.ireward_episodic,
                                                                                                trace=False)
        self.buffer['advantages'] = self.iw * self.buffer['iadvantages'] + self.ew * self.buffer['eadvantages']
        self.buffer_steps = len(self.buffer['states'])


    def compute_crange(self, result):
        steps = len(result['states'])
        cranges = np.zeros((steps, self.action_space)) + self.crange
        return cranges


    def shuffle_buffer(self):
        randomize = np.arange(self.buffer_steps)
        np.random.shuffle(randomize)
        for key in self.buffer.keys():
            #print('key', key, self.buffer[key].shape)
            self.buffer[key] = self.buffer[key][randomize]


    def create_maps(self):
        keys = ('x', 'y', 'some_map', 'ireward_map', 'entropy_map')
        self.maps = dict()
        for key in keys:
            self.maps[key] = []


    def update_maps(self, path=r'D:\sonic_models\maps.json'):
        self.create_maps()
        steps = len(self.buffer['states'])
        self.maps['x'].extend(list(self.buffer['coords'][:steps-self.cutted_steps,0]))
        self.maps['y'].extend(list(self.buffer['coords'][:steps-self.cutted_steps,1]))
        self.maps['some_map'].extend(list(self.buffer['irewards'][:steps-self.cutted_steps]))
        self.maps['ireward_map'].extend(list(self.buffer['irewards'][:steps-self.cutted_steps]))
        self.maps['entropy_map'].extend(list(self.buffer['entropy'][:steps-self.cutted_steps]))
        self.transform2pythonic(self.maps)
        try:
            with open(path, 'w') as file:
                json.dump(self.maps, file, indent=4)
        except:
            print('Error occured for saving maps')


    # def update_istats(self, result):
    #     states = self.reusable_memory[-1]['states']
    #     fast = predict([states], self.ireward_model)
    #     self.evaluate_stat_params(fast, 'fast_imean', 'fast_istd', 'fast_imax', 'fast_imin', 'fast_ivol')
    #     self.stats['initialized'] += 1

    def update_istats(self, result):
        states = result['states']
        fast = predict([states], self.ireward_model, convert_type=self.advanced_type)
        self.evaluate_stat_params(fast, 'fast_imean', 'fast_istd', 'fast_imax', 'fast_imin', 'fast_ivol')
        self.stats['initialized'] += 1


    def update_astats(self, result):
        iadvantages = result['raw_iadvantages']
        eadvantages = result['raw_eadvantages']
        iadv = np.sum(iadvantages, axis=-1)
        eadv = np.sum(eadvantages, axis=-1)
        self.evaluate_stat_params(iadv, 'iamean', 'iastd', 'iamax', 'iamin', 'iavol')
        self.evaluate_stat_params(eadv, 'eamean', 'eastd', 'eamax', 'eamin', 'eavol')


    def evaluate_stat_params(self, sequence, average, deviation, maximum, minimum, volume):
        mean = np.mean(sequence, axis=0)
        std = np.std(sequence, axis=0)
        maxi = np.max(sequence, axis=0)
        mini = np.min(sequence, axis=0)
        steps = len(sequence)
        self.stats[volume] += steps
        horizon = min(self.stats[volume], self.horizon)
        self.stats[average] = ((horizon-steps)*self.stats[average]+steps*mean)/horizon
        self.stats[deviation] = ((horizon-steps)*self.stats[deviation]+steps*std)/horizon
        self.stats[maximum] = ((horizon-steps)*self.stats[maximum]+steps*maxi)/horizon
        self.stats[minimum] = ((horizon-steps)*self.stats[minimum]+steps*mini)/horizon

    def computer(self, input_que, result_que):
        def some_func(inputs):
            rs = inputs[0]
            cvs = inputs[1]
            nvs = inputs[2]
            for ind, elem in enumerate(cvs):
                cvs[ind] = rs[ind] + self.gamma * nvs[ind]
            return cvs
        while True:
            number, *inputs = input_que.get()
            result = some_func(inputs)
            result_que.put((number, result))



    def compute_advantages(self, states, rewards, actions, value_model, gamma, lam, episodic, trace):

        rewards = rewards.astype(np.float64)
        steps = len(states)
        coeff = int(steps/self.num_computers)+1

        states = self.process_states(states)
        values = predict([states], value_model, batch_size=self.batch_size).astype(np.float64)

        # if episodic:
        #     values = np.vstack((values[:], np.zeros((1, self.action_space)) ))
        # else:
        #     values = np.vstack((values[:], values[0])).astype(np.float64)

        if episodic:
            values = np.vstack((values[:], np.zeros((1, self.action_space)) ))
        else:
            target_values = np.zeros_like(values).astype(np.float64)
            gg = np.array([self.gamma**i for i in range(steps)])
            buf = rewards[:steps]
            for i in trange(steps, ncols=130):
                s = np.sum(buf * gg)
                target_values[i] = s + self.gamma**steps * values[(i+steps)%steps]
                buf = np.hstack((buf[1:],rewards[(i+steps)%steps]))



        actions = np.hstack((actions[:], actions[0]))
        # target_values = np.copy(values).astype(np.float64)


        # for ind in trange(self.csteps, ncols=130):
        #     for j in range(self.num_computers):
        #         self.input_que.put((j,
        #                             rewards[j*coeff:(j+1)*coeff],
        #                             target_values[:-1][j*coeff:(j+1)*coeff],
        #                             target_values[1:][j*coeff:(j+1)*coeff]))
        #     results = [0] * self.num_computers
        #     for j in range(self.num_computers):
        #         number, result = self.result_que.get()
        #         results[number] = result
        #     total = np.concatenate(results)
        #     target_values[:-1] = total
        #     # for i in range(steps):
        #     #     target_values[i,actions[i]] = rewards[i] + gamma * target_values[i+1, actions[i+1]]
        #     if not episodic:
        #         target_values[-1] = target_values[0]
        # target_values = target_values[:-1]


        # target_values = np.copy(values[:])
        # actions = np.hstack((actions[:], actions[0]))
        # for ind in trange(self.csteps, ncols=130):
        #     for j in range(self.num_computers):
        #         self.input_que.put((j,
        #                             rewards[j*coeff:(j+1)*coeff],
        #                             target_values[:-1][j*coeff:(j+1)*coeff],
        #                             target_values[1:][j*coeff:(j+1)*coeff]))
        #     results = [0] * self.num_computers
        #     for j in range(self.num_computers):
        #         number, result = self.result_que.get()
        #         results[number] = result
        #     total = np.concatenate(results)
        #     target_values[:-1] = total
        #     # for i in range(steps):
        #     #     target_values[i,actions[i]] = rewards[i] + gamma * target_values[i+1, actions[i+1]]
        #     if not episodic:
        #         target_values[-1] = target_values[0]
        # target_values = target_values[:-1]

        # advantages = np.zeros((steps, self.action_space))
        # advantages_zero = np.zeros((steps+self.multi,))
        # for ind in range(steps):
        #     advantages_zero[ind] = rewards[ind] + self.gamma * values[ind+1] - values[ind]
        # advantages_zero[steps:] = advantages_zero[:self.multi]
        # for ind in range(steps):
        #     advantages[ind, actions[ind]] = np.sum(advantages_zero[ind:ind+self.multi])

        advantages = np.zeros((steps, self.action_space))
        for ind in range(steps):
            advantages[ind, actions[ind]] = target_values[ind] - values[ind]

        return advantages, target_values, np.std(target_values, axis=-1) ** 2


    def run_episode(self, policy_model, render=False, record=False, game_id=0, path='.'):
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
            cur_mem.append(np.zeros(self.state_shape, dtype=np.uint8))
        next_state = env.reset()
        while not done:
            next_state = self.resize_state(next_state)
            next_state = self.augmentate(next_state)
            next_state = np.transpose(next_state, (2, 0, 1))
            cur_mem.append(next_state)
            states.append(np.array(cur_mem, dtype=np.uint8))
            action_id, cur_policy = self.choose_action(cur_mem, policy_model, render)
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

        device = torch.device("cpu")
        if use_gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        l.acquire()
        if os.path.isfile(r'D:\sonic_models\policy_model.pt'):
            policy_model = torch.load(r'D:\sonic_models\policy_model.pt')
            policy_loading_time = os.path.getmtime(r'D:\sonic_models\policy_model.pt')
        else:
            policy_model = policy_net(self.state_shape, self.action_space)
            policy_loading_time=0
        l.release()

        while True:
            l.acquire()
            if os.path.isfile(r'D:\sonic_models\policy_model.pt'):
                if policy_loading_time<os.path.getmtime(r'D:\sonic_models\policy_model.pt'):
                    policy_loading_time = os.path.getmtime(r'D:\sonic_models\policy_model.pt')
                    try:
                        policy_model = torch.load(r'D:\sonic_models\policy_model.pt')
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
            policy_model.to(device)
            policy_model.eval()
            result = self.run_episode(policy_model, render=render, record=record_replay, game_id=game_id, path=r'D:\sonic_models\replays')
            while True:
                if data.qsize()<1:
                    data.put(result)
                    break
                else:
                    time.sleep(10)


    def choose_action(self, state_col, policy_model, render):
        state = np.reshape(np.array(state_col, dtype=np.uint8), (1, *self.state_shape))
        state = torch.from_numpy(state).to(device)
        with torch.no_grad():
            policy = policy_model(state)
        policy = policy.cpu().numpy()
        policy = policy[-1]
        order = np.random.choice(self.action_space, size=None, p=policy)
        return order, policy


    def process_states(self, states):
        states = np.array(states, dtype=np.uint8)
        states.reshape((len(states), *self.state_shape))
        return states


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

    def augmentate(self, state):
        state = Image.fromarray(state)
        if random.random()<1.5:
            state = PIL.ImageOps.invert(state)
        #state = self.aug(state)
        return np.asarray(state)


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

    def save_stats(self, path=r'D:\sonic_models\stats.json'):
        self.transform2pythonic(self.stats)
        try:
            with open(path, 'w') as file:
                json.dump(self.stats, file, indent=4)
            print('Stats saved')
        except:
            print('Error occured for saving stats')


    def update_stats(self):
        steps = len(self.buffer['entropy'])
        self.stats['action_std'].append(np.std(self.buffer['policy']))
        self.stats['steps_list'].append(steps)
        self.stats['entropy'].append(np.mean(self.buffer['entropy']))
        self.stats['external_rewards'].append(np.mean(self.buffer['erewards']))




    def reinitialize_buffer(self):
        self.buffer=dict()
        for key in self.all_keys:
            self.buffer[key]=list()


    def get_memory_length(self):
        memory_length = 0
        for result in self.reusable_memory:
            memory_length += len(result['states'])
        return memory_length


    def reduce_memory(self, by_steps=False):
        if by_steps:
            while self.get_memory_length()>self.memory_size and len(self.reusable_memory)>1:
                self.reusable_memory.pop(0)
        else:
            while len(self.reusable_memory)>20:
                self.reusable_memory.pop(0)
        gc.collect()


    def save_models(self):
        torch.save(self.policy_model, r'D:\sonic_models\policy_model.pt')
        torch.save(self.evmodel, r'D:\sonic_models\evalue_model.pt')
        torch.save(self.ivmodel, r'D:\sonic_models\ivalue_model.pt')
        torch.save(self.ireward_model, r'D:\sonic_models\ireward_model.pt')

    def save_memory(self):
        self.lock.acquire()
        print('saving memory...')
        try:
            with open(r'D:\sonic_models\memory.pickle', 'wb') as f:
                pickle.dump(self.reusable_memory, f)
            print('memory saved')
        except:
            print('unsuccess saving memory')
        finally:
            self.lock.release()

    def load_memory(self):
        self.lock.acquire()
        try:
            with open(r'D:\sonic_models\memory.pickle', 'rb') as f:
                self.reusable_memory = pickle.load(f)
            print('memory loaded succesfully')
        except:
            print('unsuccess loading memory')
            self.reusable_memory = []
        finally:
            self.lock.release()

    def create_models(self):
        if os.path.isfile(r'D:\sonic_models\policy_model.pt'):
            self.policy_model = torch.load(r'D:\sonic_models\policy_model.pt')
        else:
            self.policy_model = policy_net(self.state_shape, self.action_space)
            self.policy_model.name = 'policy_model'

        if os.path.isfile(r'D:\sonic_models\evalue_model.pt'):
            self.evmodel = torch.load(r'D:\sonic_models\evalue_model.pt')
        else:
            self.evmodel = critic_net(self.state_shape, self.action_space)
            self.evmodel.name = 'evmodel'


        if os.path.isfile(r'D:\sonic_models\ivalue_model.pt'):
            self.ivmodel = torch.load(r'D:\sonic_models\ivalue_model.pt')
        else:
            self.ivmodel = critic_net(self.state_shape, self.action_space)
            self.ivmodel.name = 'ivmodel'

        if os.path.isfile(r'D:\sonic_models\ireward_model.pt'):
            self.ireward_model = torch.load(r'D:\sonic_models\ireward_model.pt')
        else:
            self.ireward_model = reward_net(self.state_shape).to(self.advanced_type)
            self.ireward_model.name = 'ireward_model'



    def get_new_result(self, data):
        result = data.get()
        for key in self.base_keys.keys():
            result[key] = np.array(result[key], dtype=self.base_keys[key])
        return result



    def run_workers(self, lock, data, render_que, game_ids, use_gpu=False):
        processes = []
        for i in range(self.num_workers):
            processes.append(Process(target=self.worker, args=(lock, data, render_que, game_ids, use_gpu)))
        for p in processes:
            p.start()

    def run_computers(self):
        processes = []
        for i in range(self.num_computers):
            processes.append(Process(target=self.computer, args=(self.input_que, self.result_que)))
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
            self.stats['fast_ivol'] = 0

            self.stats['iamean'] = 0
            self.stats['iastd'] = 1
            self.stats['iamax'] = 1
            self.stats['iamin'] = 0
            self.stats['iavol'] = 0

            self.stats['eamean'] = 0
            self.stats['eastd'] = 1
            self.stats['eamax'] = 1
            self.stats['eamin'] = 0
            self.stats['eavol'] = 0

            self.stats['window'] = 10
            self.stats['mean100_external_rewards'] = []
            self.stats['external_rewards'] = []
            self.stats['entropy'] = []
            self.stats['action_std'] = []

            self.stats['policy_lr'] = 0.583244225269211e-00
            self.stats['vmodel_lr'] = 1e-2
            self.stats['ireward_model_lr'] = 1e-4


if __name__ == '__main__':
    agent = SonicAgent()
    agent.run_train()
    # result_file = open(r'D:\sonic_models\result.json', 'r')
    # result = json.load(result_file)
    # episode_passed = result['episode_passed']
    # policy_model = create_policy_net('policy', 1)
    # if os.path.isfile(r'D:\sonic_models\policy_model.pt'):
    #     policy_model.load_weights(r'D:\sonic_models\policy_model.pt')
    # agent.run_episode(policy_model, render=True, record=True, game_id=episode_passed, path=r'D:\sonic_models\replays')


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