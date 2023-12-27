import os
os.chdir('/content/drive/MyDrive/fall_detection/src/')

import numpy as np

from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D,
    Dense, add, MaxPool1D, Concatenate, concatenate, Add, Masking, LSTM, Dropout, Permute,
    Reshape, multiply
    )
from tensorflow.keras.models import Model


# build FCN model
def fcn_model(input_shape, nb_classes):
    input_layer = Input(input_shape)

    conv1 = Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation='relu')(conv1)

    conv2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = Conv1D(128, kernel_size=3,padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    gap_layer = GlobalAveragePooling1D()(conv3)

    output_layer = Dense(nb_classes, activation='softmax')(gap_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


# build RESNET model
def resnet_model(input_shape, nb_classes):
    n_feature_maps = 64

    input_layer = Input(input_shape)

    # BLOCK 1

    conv_x = Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_1 = add([shortcut_y, conv_z])
    output_block_1 = Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = BatchNormalization()(shortcut_y)

    output_block_2 = add([shortcut_y, conv_z])
    output_block_2 = Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = BatchNormalization()(output_block_2)

    output_block_3 = add([shortcut_y, conv_z])
    output_block_3 = Activation('relu')(output_block_3)

    # FINAL

    gap_layer = GlobalAveragePooling1D()(output_block_3)

    output_layer = Dense(nb_classes, activation='softmax')(gap_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


# define an inception block
def inception_block(
    input_tensor, stride=1, activation='linear',
    use_bottleneck=True, bottleneck_size=32,
    kernel_size=41, nb_filters=32):
    
    if use_bottleneck and int(input_tensor.shape[-1]) > bottleneck_size:
        input_inception = Conv1D(
            filters=bottleneck_size, kernel_size=1,
            padding='same', activation=activation, use_bias=False)(input_tensor)
            
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(Conv1D(
            filters=nb_filters, kernel_size=kernel_size_s[i],
            strides=stride, padding='same',
            activation=activation, use_bias=False)(input_inception))

    max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = Conv1D(
        filters=nb_filters, kernel_size=1,
        padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = Concatenate(axis=2)(conv_list)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    return x

# build a shortcut layer
def shortcut_layer(input_tensor, out_tensor):
    shortcut_y = Conv1D(
        filters=int(out_tensor.shape[-1]), kernel_size=1,
        padding='same', use_bias=False)(input_tensor)
    shortcut_y = BatchNormalization()(shortcut_y)

    x = Add()([shortcut_y, out_tensor])
    x = Activation('relu')(x)
    return x

# build INCEPTION model
def inception_model(input_shape, nb_classes, depth=6, use_residual=True):
    input_layer = Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = inception_block(x)

        if use_residual and d % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x

    gap_layer = GlobalAveragePooling1D()(x)

    output_layer = Dense(nb_classes, activation='softmax')(gap_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor

    Returns: a keras tensor
    '''
    filters = input.shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se


# build LSTMFCN model
def lstmfcn_model(input_shape, nb_classes):

    input_layer = Input(shape=(input_shape))

    x = Masking()(input_layer)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(input_layer)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    output_layer = Dense(nb_classes, activation='softmax')(x)

    model = Model(input_layer, output_layer)

    return model
