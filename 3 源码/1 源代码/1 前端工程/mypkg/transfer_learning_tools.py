from numpy.lib.shape_base import tile
from numpy.core._multiarray_umath import arange,zeros,logical_or
from numpy.lib.function_base import insert
from numpy.core.fromnumeric import cumsum

def five_to_one(this_dataset, this_offset, how_many_days, this_stock):
    # X_test/Y,test_offset,len(test_days),[each_stock]

    magic_list=tile(arange(start=5),reps=how_many_days)   # ([0,1,2,3,4],6) -> [0,1,2,3,4,0,1,2,3,4,...共*6]
    boolean_vector=zeros(shape=magic_list.shape)    # 标记准备保留的股票 (30,) -> [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    for item in this_stock:
        boolean_vector=logical_or(boolean_vector, item == magic_list) # item=0的话 [ True False False False False  True False False False False  True False False False False  True False False False False  True False False False False  True False False False False]
    left_end_point= insert(arr=this_offset, obj=0, values=0)[:-1][boolean_vector] # offset.shape==(30,), insert向最前面插入了一个0, [:-1]删掉了最后一个元素, [b...]选为True的那些个元素
    right_end_point=this_offset[boolean_vector]

    offset=cumsum(right_end_point-left_end_point)   # 往右逐个加

    rows_boolean_vector=zeros(shape=this_dataset.shape[0])
    this_row_index = arange(this_dataset.shape[0])
    for each_left,each_end in zip(left_end_point,right_end_point):
        rows_boolean_vector=logical_or(rows_boolean_vector,(each_left<=this_row_index) & (this_row_index<each_end))    # 不能是and!!!

    train=this_dataset[rows_boolean_vector]

    return [train,offset]
