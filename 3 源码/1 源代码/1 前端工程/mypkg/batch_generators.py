# Load the dataset
from numpy.lib.shape_base import expand_dims

from numpy.core._multiarray_umath import array

# Create a training subset generator for keras.engine.training.Model.fit_generator's generator to use
def batch_generator_for_RNN(normalize=False, X_dataset=None, Y_dataset=None, offset=None,sequence_length=None,batch_size=None):
        # x_test, y_test, small_test_offset，20，500
        """
        按指定的几列分割train/test集
        一个batch中仅含少等于一只股票&一天的数据(不允许多只股票/跨天)
        The words “window”, “kernel”, and “filter” are used to refer to the same thing. This is why the parameter ksize refers to “kernel size”, and we use (f,f) to refer to the filter size.

        传入参数:
        normalize -- 把每一个window中的全部行都减去第一行
        dataset -- X_train 还是 X_test
        offset -- train_offset 还是 test_offset

        :return:
        当前生成的batch
        """

        datapoint = 0
        stock_boundary = 0
        # temp_offset=[3454,9772,14694,25413,39512,44591,50713,56678,63390,77909,81812,87223,91250,96436,106444,109250,116242,121584,132403,143467,146497,153225,159171,165549,178252,180515,186758,192365,199372,217404]

        while datapoint <= (
                X_dataset.shape[0] - sequence_length):  # X_train.shape = (564978, 40) X_test.shape = (762988, 40)
            x_batch = []
            y_batch = []
            for each_batch in range(batch_size):
                # 如果到了一只股票的末尾或者数据集尾就不能再append to batch了
                if (datapoint <= X_dataset.shape[0] - sequence_length) and (
                        datapoint <= offset[stock_boundary] - sequence_length):
                    # print(type(self.offset))
                    # print(self.offset.shape)
                    # print(stock_boundary)
                    x_next_window = X_dataset[datapoint:datapoint + sequence_length]
                    y_next_window = Y_dataset[datapoint + sequence_length - 1]
                    if normalize is True:
                        x_next_window = x_next_window - x_next_window[0, :]
                    x_batch.append(x_next_window)
                    y_batch.append(y_next_window)
                    datapoint = datapoint + 1
            if datapoint > X_dataset.shape[0] - sequence_length:
                # 从数据集头来
                datapoint = 0
                stock_boundary = 0
            if datapoint > offset[stock_boundary] - sequence_length:
                # 下一只股票了,继续生成batch
                datapoint = datapoint + sequence_length - 1
                stock_boundary = stock_boundary + 1
            yield array(x_batch), array(y_batch)


def batch_generator_for_CNN(normalize=False, X_dataset=None, Y_dataset=None, offset=None, sequence_length=None,
                            batch_size=None):
    """
    按指定的几列分割train/test集
    一个batch中仅含少等于一只股票&一天的数据(不允许多只股票/跨天)
    The words “window”, “kernel”, and “filter” are used to refer to the same thing. This is why the parameter ksize refers to “kernel size”, and we use (f,f) to refer to the filter size.

    传入参数:
    normalize -- 把每一个window中的全部行都减去第一行
    dataset -- X_train 还是 X_test
    offset -- train_offset 还是 test_offset

    :return:
    当前生成的batch
    """

    datapoint = 0
    stock_boundary = 0
    # temp_offset=[3454,9772,14694,25413,39512,44591,50713,56678,63390,77909,81812,87223,91250,96436,106444,109250,116242,121584,132403,143467,146497,153225,159171,165549,178252,180515,186758,192365,199372,217404]

    while datapoint <= (
            X_dataset.shape[0] - sequence_length):  # X_train.shape = (564978, 40) X_test.shape = (762988, 40)
        x_batch = []
        y_batch = []
        for each_batch in range(batch_size):
            if (datapoint <= X_dataset.shape[0] - sequence_length) and (
                    datapoint <= offset[stock_boundary] - sequence_length):
                # print(type(self.offset))
                # print(self.offset.shape)
                # print(stock_boundary)
                x_next_window = X_dataset[datapoint:datapoint + sequence_length]
                y_next_window = Y_dataset[datapoint + sequence_length - 1]
                if normalize is True:
                    x_next_window = x_next_window - x_next_window[0, :]
                x_batch.append(x_next_window)
                y_batch.append(y_next_window)
                datapoint = datapoint + 1
        if datapoint > X_dataset.shape[0] - sequence_length:
            # 已经到文件尾，不能再append to batch了
            datapoint = 0
            stock_boundary = 0
        if datapoint > offset[stock_boundary] - sequence_length:
            # 下一只股票了,继续生成batch
            datapoint = datapoint + sequence_length - 1
            stock_boundary = stock_boundary + 1
        yield expand_dims(a=array(x_batch),axis=3), array(y_batch)

# def batch_generator_for_trans(X_dataset=None,sequence_length=None,batch_size=None,Y_dataset=None):
#     datapoint=0;
#     while datapoint <= (X_dataset.shape[0] - sequence_length):
#         x_batch = []
#         y_batch = []
#         for each_batch in range(batch_size):
#             if (datapoint <= X_dataset.shape[0] - sequence_length):
#                 x_next_window = X_dataset[datapoint:datapoint + sequence_length]
#                 y_next_window = Y_dataset[datapoint + sequence_length - 1]
#                 x_batch.append(x_next_window)
#                 y_batch.append(y_next_window)
#                 datapoint = datapoint + 1
#         if datapoint > X_dataset.shape[0] - sequence_length:
#             datapoint = 0
#         yield array(x_batch), array(y_batch)
