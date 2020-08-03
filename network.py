import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input



print('TensorFlow:', tf.__version__)


def conv_block(x, num_filters, filter_size, stride=(2,2), mode='same', act=True):
    x = Conv2D(num_filters, (filter_size, filter_size), strides=stride, padding=mode)(x)
    x = BatchNormalization()(x)
    return Activation('relu')(x) if act else x


def res_block(initial_input, num_filters=64):
    x = conv_block(initial_input, num_filters, 4, (1, 1))  # 4 instead of 3 to avoid checkerboard artifacts
    x = conv_block(x, num_filters, 4, (1, 1), act=False)  # 4 instead of 3 to avoid checkerboard artifacts
    return add([x, initial_input])

# UpSampling block - using transposed convolution
def up_block(x, upscale):
    x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=(upscale, upscale), padding='same')(x)
    x = Conv2D(filters=64, kernel_size=4, padding='same')(x)  # 4 instead of 3 to avoid checkerboard artifacts
    x = BatchNormalization()(x)
    return Activation('relu')(x)

# UpSampling block - using nearest neighbor interpolation
def up_block2(x, upscale):
    x = tf.keras.layers.UpSampling2D(size=(upscale, upscale))(x)
    x = Conv2D(filters=64, kernel_size=4, padding='same')(x)  # 4 instead of 3 to avoid checkerboard artifacts
    x = BatchNormalization()(x)
    return Activation('relu')(x)

# model
def generator(input_size=(88,88,3), upscale=4, mode='NN'):
    inp = Input(input_size)
    x = conv_block(inp, 64, 9, (1, 1))
    for i in range(4):
        x = res_block(x)
    if mode =='TS':
        x = up_block(x, upscale=int(upscale * 0.5))
        x = up_block(x, upscale=int(upscale * 0.5))
    elif mode == 'NN':
        x = up_block2(x, upscale=int(upscale * 0.5))
        x = up_block2(x, upscale=int(upscale * 0.5))
    else:
        print("please select mode TS for transposed convolution or NN for nearest neighbor interpolation as a upscaling method" )
    x = Conv2D(3, (9, 9), activation='tanh', padding='same')(x)
    outp = Lambda(lambda x: (x+1)*127.5)(x)  # this restores to range [0,255] after tanh range ([-1,1])
    return inp, outp


