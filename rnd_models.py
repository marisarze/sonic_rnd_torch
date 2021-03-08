import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import *
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.activations import tanh
import numpy as np
#from tensorflow.keras import Input


def base_net(input_shape, summary=False):
    
    activ = 'tanh'#LeakyReLU(alpha=0.3)
    def last_image(tensor):
        return tensor[:,-1,:]

    input = Input(shape=input_shape, dtype='float32')
    #float_input = K.cast(input, dtype='float32')
    float_input = Lambda(lambda input: input/255.0-0.5)(input)
    float_input = Lambda(last_image)(float_input)
    x = Conv2D(32, (8,8), activation=None)(float_input)
    x = MaxPooling2D(pool_size=(4, 4), strides=None, padding='same')(x)
    x = tanh(x)
    x = Conv2D(64, (4,4), activation=None)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
    x = tanh(x)
    x = Conv2D(64, (4,4), activation=None)(x)
    x = tanh(x)
    #x = TimeDistributed(activ)(x)
    # x = Conv2D(128, (2,2), strides=(1,1), padding='same')(x)
    # x = activ(x)
    x = Flatten()(x)
    output = Dense(512, activation='tanh')(x)
    model = Model(inputs=input, outputs=output)
    if summary:
        model.summary()
    return model

def reward_net(input_shape, summary=False):

    def difference(x):
        return K.mean(K.abs(x[0] - x[1]), axis=-1)

    def ratio(x):
        return x[0] / x[1]

    input = Input(shape=input_shape)
    fast_branch = base_net(input_shape)
    slow_branch = base_net(input_shape)
    target_branch = base_net(input_shape)
    fast_output = fast_branch(input)
    slow_output = slow_branch(input)
    target_output = target_branch(input)
    fast_output = Dense(512, activation='tanh', dtype='float64')(fast_output)
    slow_output = Dense(512, activation='tanh', dtype='float64')(slow_output)
    target_output = Dense(512, activation='tanh', dtype='float64')(target_output)
    fast_part = Model(inputs=input, outputs=fast_output)
    slow_part = Model(inputs=input, outputs=slow_output)
    target_part = Model(inputs=input, outputs=target_output)
    for layer in target_part.layers:
        layer.trainable = False

    fast_loss = Lambda(difference)([target_output, fast_output])
    slow_loss = Lambda(difference)([target_output, slow_output])
    ratio = Lambda(ratio)([fast_loss, slow_loss])
    model = Model(inputs=input, outputs=[fast_loss, slow_loss, ratio])
    
    if summary:
        model.summary()
    return model

def policy_net(input_shape, action_space, summary=False):
    state_input = Input(shape=input_shape)
    conv_part = base_net(input_shape, action_space)
    main_output = Dense(action_space, activation='softmax', dtype='float64')(conv_part(state_input))
    model = Model(inputs=state_input, outputs=main_output)
    if summary:
        model.summary()
    return model

def critic_net(input_shape, epsilon, summary=False):
    state_input = Input(shape=input_shape)
    conv_part = base_net(input_shape)
    x = conv_part(state_input)
    x = Dense(512, activation='tanh', dtype='float64')(x)
    critic_output = Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.Zeros())(x)
    model = Model(inputs=state_input, outputs=critic_output)

    weights = [np.array(w) for w in model.get_weights()]
    weights[-1] *= 0
    model.set_weights(weights)

    if summary:
        model.summary()
    return model

if __name__ == "__main__":
    import numpy as np
    width = 120
    height = 84
    state_shape = (1,height,width,3)
    net = reward_net(state_shape)
    states = np.random.rand(1,*state_shape)
    
    preloss = net.predict(states)
    print(preloss)

    