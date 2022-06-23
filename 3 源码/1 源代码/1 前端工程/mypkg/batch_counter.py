# Training
from numpy.lib.function_base import diff
from numpy.core.fromnumeric import sum
from numpy.core._multiarray_umath import ceil

from numpy.lib.function_base import insert

# The entire training dataset are huge in size and can't fit into my computer's RAM
def batches_amount_per_epoch(offset, sequence_length, batch_size):    # how_many_batches_to_yield_per_epoch
    """
    "The generator is expected to loop over its data indefinitely."
    这个函数计算了要过完给定的数据集一遍，生成器要生成几次batch

    :param offset: 是train_offest还是test_offset
    :param sequence_length:
    :param batch_size:

    :return:
    """
    difference = diff(insert(arr=offset, obj=0, values=0))  # 某一只股票的数据量，连续指定天
    epoch_size = sum([ceil((number - sequence_length) / batch_size) for number in difference])

    return epoch_size
