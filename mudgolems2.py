import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from tqdm import trange


def evaluate(inputs, model, loss_fn, batch_size=32, verbose=True, **kwargs):
    all_len = 0 
    for input in inputs:
        if hasattr(input, "__len__"):
            all_len = max(len(input), all_len)
    nsteps = all_len // batch_size
    if all_len % batch_size:
        nsteps += 1
    accumulated_loss = 0
    pbar = trange(nsteps, ncols=150, disable=not verbose)
    for step in pbar:
        start = batch_size * step
        end = min(start + batch_size, all_len)
        delta = end-start
        batch_input = [input[start:end] for input in inputs]
        with tf.GradientTape() as tape:
            batch_loss = loss_fn(batch_input, model, **kwargs)
        accumulated_loss += batch_loss.numpy()
        mean_loss = accumulated_loss/end
        pbar.set_description('Evaluating {} '.format(model.name))
        pbar.set_postfix_str('mean_loss: {:.6E}'.format(mean_loss))
    return mean_loss


def train_by_batch(inputs, model, loss_fn, optimizer=None, batch_size=32, epochs=1, verbose=True, **kwargs):
    loss_history = []
    
    if not optimizer:
        optimizer = model.optimizer
    for epoch in range(epochs):
        all_len = 0 
        for input in inputs:
            if hasattr(input, "__len__"):
                all_len = max(len(input), all_len)
        epoch_nsteps = all_len // batch_size
        if all_len % batch_size:
            epoch_nsteps += 1
        accumulated_epoch_loss = 0
        pbar = trange(epoch_nsteps, ncols=150, disable=not verbose)
        for step in pbar:
            start = batch_size * step
            end = min(start + batch_size, all_len)
            delta = end-start
            batch_input = [input[start:end] for input in inputs]
            loss = batch_train_step(batch_input, model, loss_fn, optimizer, **kwargs) 
            accumulated_epoch_loss += loss.numpy()
            mean_epoch_loss = accumulated_epoch_loss/end
            pbar.set_description('Training {}, epoch {}/{}'.format(model.name, epoch+1, epochs))
            pbar.set_postfix_str('mean_loss: {:.6E}'.format(mean_epoch_loss))
        loss_history.append(accumulated_epoch_loss / all_len)
    model.compile(optimizer=optimizer)
    return loss_history

def train_by_epoch(inputs, model, loss_fn, optimizer=None, batch_size=32, epochs=1, verbose=True, **kwargs):
    loss_history = []
    training_weights = model.trainable_variables
    if not optimizer:
        optimizer = model.optimizer
    for epoch in range(epochs):
        all_len = 0 
        for input in inputs:
            if hasattr(input, "__len__"):
                all_len = max(len(input), all_len)
        epoch_nsteps = all_len // batch_size
        if all_len % batch_size:
            epoch_nsteps += 1
        accumulated_epoch_loss = 0
        accumulated_grads = [tf.zeros_like(layer) for layer in training_weights]
        pbar = trange(epoch_nsteps, ncols=150, disable=not verbose)
        for step in pbar:
            start = batch_size * step
            end = min(start + batch_size, all_len)
            delta = end-start
            batch_input = [input[start:end] for input in inputs]
            loss, grads = grad_tape(batch_input, model, loss_fn, **kwargs)
            accumulated_epoch_loss += loss.numpy()
            mean_epoch_loss = accumulated_epoch_loss/end
            for i, grad in enumerate(grads):
                if grad is not None:
                    accumulated_grads[i] += grad
            pbar.set_description('Training {}, epoch {}/{}'.format(model.name, epoch+1, epochs))
            pbar.set_postfix_str('mean_loss: {:.6E}'.format(mean_epoch_loss))
        accumulated_grads = [gradient/all_len for gradient in accumulated_grads]
        optimizer.apply_gradients(zip(accumulated_grads, training_weights))
        loss_history.append(accumulated_epoch_loss / all_len)
    model.compile(optimizer=optimizer)
    return loss_history



def batch_train_step(inputs, model, loss_fn, optimizer, **kwargs):
    input_len = len(inputs[0])
    initial_grads = [tf.zeros_like(layer) for layer in model.trainable_variables]
    loss, grads = grad_tape(inputs, model, loss_fn, **kwargs)
    for i, grad in enumerate(grads):
        if grad is not None:
            initial_grads[i] += grad / input_len
    optimizer.apply_gradients(zip(initial_grads, model.trainable_variables))
    return loss

#@tf.function
def grad_tape(inputs, model, loss_fn, **kwargs):
    with tf.GradientTape() as tape:
        loss = loss_fn(inputs, model, **kwargs)
    grads = tape.gradient(loss, model.trainable_variables)
    return loss, grads



if __name__ == "__main__":
    from rnd_models import *

    def policy_net(input_shape, action_space, summary=False):
        reward_input = Input(shape=(action_space,))
        state_input = Input(shape=input_shape)
        conv_part = base_net(input_shape, action_space)
        main_output = Dense(action_space, activation='softmax')(conv_part(state_input))
        model = Model(inputs=state_input, outputs=main_output, name='Chunga')
        return model

    def my_loss_fn(inputs, model, **kwargs):
        states = inputs[0]
        rewards = inputs[1]
        old_policies = inputs[2]
        crange = inputs[3]
        policy = model(states)
        beta= 1/2/crange
        base_loss = K.sum(-rewards * policy, axis=-1)
        inertia_loss = beta *  K.sum(K.abs(rewards), axis=-1) * K.sum(K.pow(policy-old_policies, 2), axis=-1)/tf.constant(action_space, tf.float32)
        loss = tf.reduce_sum(base_loss + inertia_loss)
        return loss


    steps = 50000
    width = 120
    height = 84
    action_space = 10
    state_shape = (1,height,width,3)
    net = policy_net(state_shape, action_space)
    states = tf.constant(np.random.rand(steps,*state_shape), dtype=tf.float32)#.astype(np.float32)
    rewards = tf.constant(np.random.rand(steps, action_space), dtype=tf.float32)#.astype(np.float32)
    crange = tf.constant(np.zeros((steps, 1)) + 0.05, dtype=tf.float32)
    #crange = crange.astype(np.float32)
    old_policies = tf.constant(net.predict(states), dtype=tf.float32)
    inputs = [states, rewards, old_policies, crange]
    optimizer = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-7, clipnorm=1.0)
    losses = train_by_epoch(inputs, net, my_loss_fn, optimizer, 200, 10)
    print(losses)
    # some_loss = my_loss_fn(inputs, net)
    # print(len(some_loss))



    

























# for layer in model.layers[::-1]:
#     print('--------------------') 
#     # print(layer.get_config())
#     # print(layer.weights)
#     print(layer, layer._inbound_nodes[0].inbound_layers)