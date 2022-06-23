# Implement the RNN model
from keras.engine.input_layer import Input
# from keras.layers.convolutional import Conv2D
from keras.layers.recurrent import LSTM
from keras.regularizers import l2   # Add L2 正则化
from keras.layers.core import Dense, Dropout
# from keras.layers.pooling import MaxPooling2D
from keras.engine.training import Model

# Implement the CNN model
from keras.engine.input_layer import Input
from keras.layers.convolutional import Conv2D,Conv1D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.pooling import MaxPooling2D
from keras.engine.training import Model

# Implement the CNN-RNN model
from keras.engine.sequential import Sequential
from keras.backend.tensorflow_backend import squeeze
from keras.layers.core import Lambda

# Implement the 1st RNN model

def RNN_first_try(input_shape):
    """
    第一次搭建RNN的尝试:
    LSTM + L2 -> Dropout -> FullyConnected + SoftMax

    传入参数:
    input_shape -- 数据集形状

    :return:
    model_RNN -- Keras 中 Model() 的一个 instance
    """

    # 模型输入为 input_shape 大小的 tensor
    X_input = Input(shape=input_shape)

    # LSTM layer
    X = LSTM(units=100,kernel_regularizer=l2())(X_input)

    # Dropout
    X = Dropout(rate=0.5)(X)

    # FullyConnected + SoftMax
    X = Dense(units=3, activation='softmax')(X)

    # Create a Keras model instance
    model_RNN = Model(inputs=X_input, outputs=X, name='RNN_first_try')

    return model_RNN

# Implement the 1st CNN2D model

def Conv2D_1st_try(input_shape):
    """
    第一次搭建CNN的尝试:
    Conv2D -> RELU -> Conv2D -> RELU -> MaxPool -> Conv2D -> RELU -> MaxPool -> Conv2D -> RELU -> MaxPool -> Flatten + FullyConnected + RELU -> FullyConnected + SoftMax

    传入参数:
    input_shape -- 数据集形状

    :return:
    model_CNN -- Keras 中 Model() 的一个 instance
    """

    # 模型输入为 input_shape 大小的 tensor
    X_input = Input(shape=input_shape)

    # Conv2D -> RELU
    X = Conv2D(filters=16, kernel_size=(4, 40), strides=(1, 1), data_format='channels_last')(X_input)
    X = Activation('relu')(X)

    # Conv2D -> RELU
    X = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), data_format='channels_last')(X)
    X = Activation('relu')(X)

    # MaxPool
    X = MaxPooling2D(pool_size=(2, 1), strides=2)(X)

    # Conv2D -> RELU
    X = Conv2D(filters=32, kernel_size=(4, 1), strides=(1, 1), data_format='channels_last')(X)
    X = Activation('relu')(X)

    # MaxPool
    X = MaxPooling2D(pool_size=(2, 1), strides=2)(X)

    # Conv2D -> RELU
    X = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), data_format='channels_last')(X)
    X = Activation('relu')(X)

    # MaxPool
    X = MaxPooling2D(pool_size=(2, 1), strides=2)(X)

    # Output Layer: convert X to a vector + FullyConnected + RELU
    X = Flatten()(X)
    X = Dense(units=64, activation='relu')(X)

    # FullyConnected + SoftMax
    X = Dense(units=3, activation='softmax')(X)

    # Create a Keras model instance
    model_CNN = Model(inputs=X_input, outputs=X, name='Conv2D_1st_try')

    return model_CNN

# Implement the 2nd CNN2D model

def Conv2D_2nd_try(input_shape):
    """
    第二次搭建CNN的尝试:
    Conv2D -> RELU -> Conv2D -> RELU -> (MaxPool) -> Conv2D -> RELU -> MaxPool -> Conv2D -> RELU -> MaxPool -> Flatten + FullyConnected↑ + RELU -> +60% Dropout-> FullyConnected + SoftMax

    传入参数:
    input_shape -- 数据集形状

    :return:
    model_CNN -- Keras 中 Model() 的一个 instance
    """

    # 模型输入为 input_shape 大小的 tensor
    X_input = Input(shape=input_shape)

    # Conv2D -> RELU
    X = Conv2D(filters=16, kernel_size=(4, 40), strides=(1, 1), data_format='channels_last')(X_input)
    X = Activation('relu')(X)

    # Conv2D -> RELU
    X = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), data_format='channels_last')(X)
    X = Activation('relu')(X)

    # (MaxPool)
    # X = MaxPooling2D(pool_size=(2, 1), strides=2)(X)

    # Conv2D -> RELU
    X = Conv2D(filters=32, kernel_size=(4, 1), strides=(1, 1), data_format='channels_last')(X)
    X = Activation('relu')(X)

    # MaxPool
    X = MaxPooling2D(pool_size=(2, 1), strides=2)(X)

    # Conv2D -> RELU
    X = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), data_format='channels_last')(X)
    X = Activation('relu')(X)

    # MaxPool
    X = MaxPooling2D(pool_size=(2, 1), strides=2)(X)

    # Output Layer: convert X to a vector + FullyConnected + RELU
    X = Flatten()(X)
    X = Dense(units=100, activation='relu')(X)

    # 60% Dropout
    X = Dropout(rate=0.6)(X)

    # FullyConnected + SoftMax
    X = Dense(units=3, activation='softmax')(X)

    # Create a Keras model instance
    model_CNN = Model(inputs=X_input, outputs=X, name='Conv2D_2nd_try')

    return model_CNN

# Implement the 3rd CNN2D model

def Conv2D_3rd_try(input_shape):
    """
    第三次搭建CNN的尝试:
    Conv2D -> RELU -> Conv2D -> RELU -> Conv2D -> RELU -> MaxPool -> Conv2D -> RELU -> MaxPool -> Flatten + FullyConnected(l2) + RELU -> 50% Dropout↓-> FullyConnected + SoftMax

    传入参数:
    input_shape -- 数据集形状

    :return:
    model_CNN -- Keras 中 Model() 的一个 instance
    """

    # 模型输入为 input_shape 大小的 tensor
    X_input = Input(shape=input_shape)

    # Conv2D -> RELU
    X = Conv2D(filters=16, kernel_size=(4, 40), strides=(1, 1), data_format='channels_last')(X_input)
    X = Activation('relu')(X)

    # Conv2D -> RELU
    X = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), data_format='channels_last')(X)
    X = Activation('relu')(X)

    # Conv2D -> RELU
    X = Conv2D(filters=32, kernel_size=(4, 1), strides=(1, 1), data_format='channels_last')(X)
    X = Activation('relu')(X)

    # MaxPool
    X = MaxPooling2D(pool_size=(2, 1), strides=2)(X)

    # Conv2D -> RELU
    X = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), data_format='channels_last')(X)
    X = Activation('relu')(X)

    # MaxPool
    X = MaxPooling2D(pool_size=(2, 1), strides=2)(X)

    # Output Layer: convert X to a vector + FullyConnected + RELU
    X = Flatten()(X)
    X = Dense(units=100, kernel_regularizer=l2(),activation='relu')(X)

    # 50% Dropout
    X = Dropout(rate=0.5)(X)

    # FullyConnected + SoftMax
    X = Dense(units=3, activation='softmax')(X)

    # Create a Keras model instance
    model_CNN = Model(inputs=X_input, outputs=X, name='Conv2D_3rd_try')

    return model_CNN

# Implement the 1st CNN1D model

def Conv1D_1st_try(input_shape):
    """
    第一次搭建CNN1D的尝试:
    Conv1D -> RELU -> Conv1D(with dilation) -> RELU -> Conv1D(with dilation) -> RELU -> Conv1D(with dilation) -> RELU -> Conv1D(with dilation) -> RELU -> Flatten + FullyConnected(l2) + RELU -> 40% Dropout-> FullyConnected + SoftMax

    传入参数:
    input_shape -- 数据集形状

    :return:
    model_CNN -- Keras 中 Model() 的一个 instance
    """

    # 模型输入为 input_shape 大小的 tensor
    X_input = Input(shape=input_shape)

    # Conv1D -> RELU
    X = Conv1D(filters=16, kernel_size=2, padding='causal', strides=1)(X_input)
    X = Activation('relu')(X)

    # Conv1D(with dilation) -> RELU
    X = Conv1D(filters=16, kernel_size=2, padding='causal', dilation_rate=2)(X)
    X = Activation('relu')(X)

    # Conv1D(with dilation) -> RELU
    X = Conv1D(filters=16, kernel_size=2, padding='causal', dilation_rate=4)(X)
    X = Activation('relu')(X)

    # Conv1D(with dilation) -> RELU
    X = Conv1D(filters=16, kernel_size=2, padding='causal', dilation_rate=8)(X)
    X = Activation('relu')(X)

    # Conv1D(with dilation) -> RELU
    X = Conv1D(filters=16, kernel_size=2, padding='causal', dilation_rate=16)(X)
    X = Activation('relu')(X)

    # Output Layer: convert X to a vector + FullyConnected + RELU
    X = Flatten()(X)
    X = Dense(units=64, kernel_regularizer=l2(),activation='relu')(X)

    # 40% Dropout
    X = Dropout(rate=0.4)(X)

    # FullyConnected + SoftMax
    X = Dense(units=3, activation='softmax')(X)

    # Create a Keras model instance
    model_CNN_1D = Model(inputs=X_input, outputs=X, name='Conv1D_1st_try')

    return model_CNN_1D

# Implement the 1st CNN-RNN model

def squeeze_axis(x):
    """
    去掉三维通道的最后一维(CNN通道上面定义为了channel-last)
    :param x:
    :return:
    """

    return squeeze(x=x,axis=2)

def CNN_RNN_1st_try():
    """
    第一次搭建CNN+RNN的尝试:
    Conv2D -> RELU -> Conv2D -> RELU -> Conv2D -> RELU -> MaxPool -> Conv2D -> RELU -> MaxPool -> LSTM + RELU -> 50% Dropout-> FullyConnected + SoftMax

    传入参数:
    input_shape -- 数据集形状

    :return:
    model_CNN -- Keras 中 Model() 的一个 instance
    """

    # 模型输入为 input_shape 大小的 tensor
    # X_input = Input(shape=input_shape)

    model_CNN_RNN=Sequential()

    # Conv2D -> RELU

    # X = Conv2D(filters=16, kernel_size=(4, 40), strides=(1, 1), data_format='channels_last')(X_input)
    # X = Activation('relu')(X)
    model_CNN_RNN.add(layer=Conv2D(input_shape=(100,40,1), filters=16, kernel_size=(4, 40), strides=(1, 1), data_format='channels_last',activation='relu'))

    # Conv2D -> RELU
    # X = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), data_format='channels_last')(X)
    # X = Activation('relu')(X)
    model_CNN_RNN.add(layer=Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), data_format='channels_last',activation='relu'))

    # Conv2D -> RELU
    # X = Conv2D(filters=32, kernel_size=(4, 1), strides=(1, 1), data_format='channels_last')(X)
    # X = Activation('relu')(X)
    model_CNN_RNN.add(layer=Conv2D(filters=32, kernel_size=(4, 1), strides=(1, 1), data_format='channels_last',activation='relu'))

    # MaxPool
    # X = MaxPooling2D(pool_size=(2, 1), strides=2)(X)
    model_CNN_RNN.add(layer=MaxPooling2D(pool_size=(2, 1), strides=2))

    # Conv2D -> RELU
    # X = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), data_format='channels_last')(X)
    # X = Activation('relu')(X)
    model_CNN_RNN.add(layer=Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), data_format='channels_last',activation='relu'))

    # MaxPool
    # X = MaxPooling2D(pool_size=(2, 1), strides=2)(X)
    model_CNN_RNN.add(layer=MaxPooling2D(pool_size=(2, 1), strides=2))

    # Squeeze channel dimension
    model_CNN_RNN.add(layer=Lambda(function=squeeze_axis))

    # LSTM + RELU
    # X = Flatten()(X)
    # X = Dense(units=100, kernel_regularizer=l2(),activation='relu')(X)
    # X = LSTM(units=100,kernel_regularizer=l2())(X)
    # X = Activation('relu')(X)
    model_CNN_RNN.add(layer=LSTM(units=100,kernel_regularizer=l2(),activation='relu'))

    # 50% Dropout
    # X = Dropout(rate=0.5)
    model_CNN_RNN.add(layer=Dropout(rate=0.5))

    # FullyConnected + SoftMax
    # X = Dense(units=3, activation='softmax')
    model_CNN_RNN.add(layer=Dense(units=3, activation='softmax'))

    # Create a Keras model instance
    # model_CNN = Model(inputs=X_input, outputs=X, name='CNN_RNN_1st_try')

    return model_CNN_RNN

# Implement the 2nd CNN-RNN model

def CNN_RNN_2nd_try():
    """
    第二次搭建CNN+RNN的尝试:
    Conv2D -> RELU -> Conv2D -> RELU -> Conv2D -> RELU -> MaxPool -> Conv2D -> RELU -> MaxPool -> LSTM + RELU -> 60% Dropout↑-> FullyConnected + SoftMax

    传入参数:
    input_shape -- 数据集形状

    :return:
    model_CNN -- Keras 中 Model() 的一个 instance
    """

    # 模型输入为 input_shape 大小的 tensor
    # X_input = Input(shape=input_shape)

    model_CNN_RNN = Sequential()

    # Conv2D -> RELU

    # X = Conv2D(filters=16, kernel_size=(4, 40), strides=(1, 1), data_format='channels_last')(X_input)
    # X = Activation('relu')(X)
    model_CNN_RNN.add(layer=Conv2D(input_shape=(100, 40, 1), filters=16, kernel_size=(4, 40), strides=(1, 1),
                                   data_format='channels_last', activation='relu'))

    # Conv2D -> RELU
    # X = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), data_format='channels_last')(X)
    # X = Activation('relu')(X)
    model_CNN_RNN.add(
        layer=Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), data_format='channels_last', activation='relu'))

    # Conv2D -> RELU
    # X = Conv2D(filters=32, kernel_size=(4, 1), strides=(1, 1), data_format='channels_last')(X)
    # X = Activation('relu')(X)
    model_CNN_RNN.add(
        layer=Conv2D(filters=32, kernel_size=(4, 1), strides=(1, 1), data_format='channels_last', activation='relu'))

    # MaxPool
    # X = MaxPooling2D(pool_size=(2, 1), strides=2)(X)
    model_CNN_RNN.add(layer=MaxPooling2D(pool_size=(2, 1), strides=2))

    # Conv2D -> RELU
    # X = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), data_format='channels_last')(X)
    # X = Activation('relu')(X)
    model_CNN_RNN.add(
        layer=Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), data_format='channels_last', activation='relu'))

    # MaxPool
    # X = MaxPooling2D(pool_size=(2, 1), strides=2)(X)
    model_CNN_RNN.add(layer=MaxPooling2D(pool_size=(2, 1), strides=2))

    # Squeeze channel dimension
    model_CNN_RNN.add(layer=Lambda(function=squeeze_axis))

    # LSTM + RELU
    # X = Flatten()(X)
    # X = Dense(units=100, kernel_regularizer=l2(),activation='relu')(X)
    # X = LSTM(units=100,kernel_regularizer=l2())(X)
    # X = Activation('relu')(X)
    model_CNN_RNN.add(layer=LSTM(units=100, kernel_regularizer=l2(), activation='relu'))

    # 60% Dropout
    # X = Dropout(rate=0.5)
    model_CNN_RNN.add(layer=Dropout(rate=0.6))

    # FullyConnected + SoftMax
    # X = Dense(units=3, activation='softmax')
    model_CNN_RNN.add(layer=Dense(units=3, activation='softmax'))

    # Create a Keras model instance
    # model_CNN = Model(inputs=X_input, outputs=X, name='CNN_RNN_2nd_try')

    return model_CNN_RNN
