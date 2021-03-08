import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from tqdm import trange, tqdm


def evaluate_tf(inputs, model, loss_fn, batch_size=32, verbose=True, **kwargs):
    steps = len(inputs[0]) // batch_size if len(inputs[0]) % batch_size else len(inputs[0]) // batch_size + 1
    dataset = tf.data.Dataset.zip(tuple(tf.data.Dataset.from_tensor_slices(item) for item in inputs))
    dataset = dataset.batch(batch_size)
    accumulated_epoch_loss = 0
    processed_samples = 0
    pbar = tqdm(dataset, total=steps, ncols=130, disable=not verbose)
    for batch in pbar:
        losses = loss_fn(batch, model, **kwargs)
        processed_samples += len(losses)
        accumulated_loss += tf.reduce_sum(losses).numpy()
        mean_loss = accumulated_loss/processed_samples
        pbar.set_description('Evaluating {} '.format(model.name))
        pbar.set_postfix_str('mean_loss: {:.6E}'.format(mean_loss))
    return mean_loss


def train_by_batch_tf(inputs, model, loss_fn, optimizer=None, batch_size=32, epochs=1, verbose=True, **kwargs):
    loss_history = []
    steps = len(inputs[0]) // batch_size if len(inputs[0]) % batch_size else len(inputs[0]) // batch_size + 1 
    if not optimizer:
        optimizer = model.optimizer
    for epoch in range(epochs):
        dataset = tf.data.Dataset.zip(tuple(tf.data.Dataset.from_tensor_slices(item) for item in inputs))
        dataset = dataset.batch(batch_size)
        accumulated_epoch_loss = 0
        processed_samples = 0
        pbar = tqdm(dataset, total=steps, ncols=130, disable=not verbose)
        for batch in pbar:
            with tf.GradientTape() as tape:
                losses = loss_fn(batch, model, **kwargs)
                loss = tf.reduce_sum(losses)
                grads = tape.gradient(tf.reduce_mean(losses), model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            processed_samples += len(losses) 
            accumulated_epoch_loss += loss.numpy()
            mean_epoch_loss = accumulated_epoch_loss/processed_samples
            pbar.set_description('Training {}, epoch {}/{}'.format(model.name, epoch+1, epochs))
            pbar.set_postfix_str('mean_loss: {:.6E}'.format(mean_epoch_loss))
        loss_history.append(accumulated_epoch_loss / processed_samples)
    model.compile(optimizer=optimizer)
    return loss_history

def train_by_epoch_tf(inputs, model, loss_fn, optimizer=None, batch_size=32, epochs=1, verbose=True, **kwargs):
    loss_history = []
    steps = len(inputs[0]) // batch_size if len(inputs[0]) % batch_size else len(inputs[0]) // batch_size + 1
    training_weights = model.trainable_variables
    if not optimizer:
        optimizer = model.optimizer
    for epoch in range(epochs):
        dataset = tf.data.Dataset.zip(tuple(tf.data.Dataset.from_tensor_slices(item) for item in inputs))
        dataset = dataset.batch(batch_size)
        accumulated_epoch_loss = 0
        processed_samples = 0
        accumulated_grads = [tf.zeros_like(layer) for layer in training_weights]
        pbar = tqdm(dataset, total=steps, ncols=130, disable=not verbose)
        for batch in pbar:
            with tf.GradientTape() as tape:
                losses = loss_fn(batch, model, **kwargs)
                loss = tf.reduce_sum(losses)
                grads = tape.gradient(loss, model.trainable_variables)
            processed_samples += len(losses)
            accumulated_epoch_loss += loss.numpy()
            mean_epoch_loss = accumulated_epoch_loss / processed_samples
            for i, grad in enumerate(grads):
                if grad is not None:
                    accumulated_grads[i] += grad
            pbar.set_description('Training {}, epoch {}/{}'.format(model.name, epoch+1, epochs))
            pbar.set_postfix_str('mean_loss: {:.6E}'.format(mean_epoch_loss))
        accumulated_grads = [gradient/processed_samples for gradient in accumulated_grads]
        optimizer.apply_gradients(zip(accumulated_grads, training_weights))
        loss_history.append(accumulated_epoch_loss / processed_samples)
    model.compile(optimizer=optimizer)
    return loss_history


