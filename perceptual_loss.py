from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input


# get VGG network
def get_VGG16(input_size):
    vgg_inp = Input(input_size)
    vgg = VGG16(include_top=False, input_tensor=vgg_inp)
    for l in vgg.layers:
        l.trainable = False
    vgg_outp = vgg.get_layer('block2_conv2').output

    return vgg_inp, vgg_outp


