import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Lambda, LeakyReLU, Conv2D, Flatten, Input, TimeDistributed, LSTM, Layer
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import Model

input0 = Input(shape=(1,))
x = Dense(10)(input0)
x = Dense(10)(x)

tmodel = Model(inputs=input0, outputs=x)

input1 = Input(shape=(1,))
input2 = Input(shape=(1,))


class temp_layer(Layer):

  def __init__(self):
      super(temp_layer, self).__init__()

  def call(self, inputs):  # Defines the computation from inputs to outputs
      return tmodel(inputs)

alayer = temp_layer()
r1 = alayer(input1)
r2 = alayer(input2)

print(alayer.get_output_at(0))
print(alayer.get_output_at(1))

