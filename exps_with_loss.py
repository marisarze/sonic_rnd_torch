import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, models

# disabling eager execution makes this example work:
# tf.python.framework_ops.disable_eager_execution()


def get_loss_fcn(w):
    def loss_fcn(y_true, y_pred):
        loss = w * losses.mse(y_true, y_pred)
        return loss
    return loss_fcn


data_x = np.random.rand(500, 4, 1)
data_w = np.random.rand(500, 4)
data_y = np.random.rand(500, 4, 1)

x = layers.Input([4, 1])
w = layers.Input([4])
y = layers.Activation('tanh')(x)
model = models.Model(inputs=[x, w], outputs=y)
loss = get_loss_fcn(w)

# using another loss makes it work, too:
# loss = 'mse'

model.compile(loss=loss)
model.fit([data_x, data_w], data_y, batch_size=1)

print('Done.')