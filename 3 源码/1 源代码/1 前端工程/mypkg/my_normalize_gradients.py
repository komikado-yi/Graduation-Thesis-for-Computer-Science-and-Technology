from keras.backend.tensorflow_backend import sqrt,mean,square
from keras.backend.common import epsilon

def normalize(x):
    """
    没有减均值版的标准化，可以使梯度不会过小或过大，让梯度上升平滑进行
    :param x:
    :return:normalized_x
    """

    normalized_x=x / (sqrt(x=mean(x=square(x=x))) + epsilon())

    return normalized_x
