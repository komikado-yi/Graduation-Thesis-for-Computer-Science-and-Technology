# Generate .pkl files from dataset
from zipfile import ZipFile
from io import StringIO
from pandas.io.parsers import read_csv
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame
from pandas import concat
from numpy.core._multiarray_umath import array
from numpy.lib.function_base import insert,diff
from pandas import DataFrame
from numpy.core.fromnumeric import repeat

# 训练集的dataframe，返回的是converted_dataframe_list
def convert_multiple_days_data_to_a_dataframe(selected_train_or_test_days):
    """
    把多天数据生成[dataframe]

    传入参数:
    selected_train_or_test_days -- 训练/测试集所含哪些天

    :return:
    converted_dataframe_list -- 包含多天数据的[dataframe]
    """

    all_staged_dataset=[f'Day{num}.zip' for num in range(1,10+1)]
    file_name_in_zip=[f'Test_Dst_NoAuction_ZScore_CF_{ordinal}.txt' for ordinal in range(1,9+1)]
    file_name_in_zip.insert(0,'Train_Dst_NoAuction_ZScore_CF_1.txt')

    converted_dataframe_list=[]

    for each_day in selected_train_or_test_days:
        loaded_1_day_zip=ZipFile(file='dataset/'+all_staged_dataset[each_day],mode='r') # 读这天的这个zip
        the_data_in_zip=StringIO(initial_value=loaded_1_day_zip.read(name=file_name_in_zip[each_day]).decode(encoding='utf-8'))    # 读压缩包里指定天的Test文件
        converted_dataframe=read_csv(filepath_or_buffer=the_data_in_zip,sep='\s{2,}',header=None,engine='python')
        converted_dataframe=DataFrame(data=converted_dataframe.values.T)    # (149,xxxxx)   ->  (xxxxx,149)
        converted_dataframe_list.append(converted_dataframe)

    converted_dataframe_list=concat(objs=converted_dataframe_list)  # 把list中的每一个小list无缝拼接在一起

    return converted_dataframe_list

# 训练集的偏移，见数据集论文Fig.2，返回的是vector_of_stock_offest
def offset_for_each_stock_each_day(selected_train_or_test_days):
    """
    用来确定每一个原始数据集文件中「当天」「各只股票」的entries有多少条
    这样生成的每一个batch中就只含某一天中的一只股票了

    传入参数:
    selected_train_or_test_days -- 训练/测试集所含哪些天

    :return:
    vector_of_stock_offest -- 第一天1～5各支股票的量，第二天1～5各支股票的量（在文件中的位置）
    """

    offsets=array([[0, 3454,  9772, 14694, 25413, 39512],   # 每一行是一天，每两个数之差代表某一只股票的数据量
                    [0, 5079, 11201, 17166, 23878, 38397],
                    [0, 3903,  9314, 13341, 18527, 28535],
                    [0, 2806,  9798, 15140, 25959, 37023],
                    [0, 3030,  9758, 15704, 22082, 34785],
                    [0, 2263,  8506, 14113, 21120, 39152],
                    [0, 2801,  9861, 16601, 24455, 37346],
                    [0, 2647, 11309, 19900, 33129, 55478],
                    [0, 1873, 11144, 21180, 34060, 52172],
                    [0, 1888,  7016, 12738, 18559, 31937]])

    temp1=offsets[selected_train_or_test_days,1:]  # 选指定的那几行，并且顺便把一天中开始的0(第一列)全都删掉
    temp2=offsets[selected_train_or_test_days,-1].cumsum() # 对最后一列进行逐步求和，会得到前x天共ndarray中第x-1  位置上的数条数据。如前六天就会返回一个大小为(6,)的ndarray:[ 39512  77909 106444 143467 178252 217404]
    temp2=insert(arr=temp2,obj=0,values=0)[:-1].reshape(-1,1) # 往最前面插个0并把最后一个元素删掉,再弄成(6, 1)形状的ndarray
    vector_of_stock_offset=(temp1+temp2).reshape(-1)    # 给temp1的每一列都加上面(6, 1)形状的ndarray:[[     0]\r[ 39512]\r[ 77909]\r[106444]\r[143467]\r[178252]],最后拉平成(5*6=30,)

    return vector_of_stock_offset

def my_repeated_index(days):
    """

    :param days:
    :return:
    """
    offsets = array([[0, 3454, 9772, 14694, 25413, 39512],  # 每一行是一天，每两个数之差代表某一只股票的数据量
                     [0, 5079, 11201, 17166, 23878, 38397],
                     [0, 3903, 9314, 13341, 18527, 28535],
                     [0, 2806, 9798, 15140, 25959, 37023],
                     [0, 3030, 9758, 15704, 22082, 34785],
                     [0, 2263, 8506, 14113, 21120, 39152],
                     [0, 2801, 9861, 16601, 24455, 37346],
                     [0, 2647, 11309, 19900, 33129, 55478],
                     [0, 1873, 11144, 21180, 34060, 52172],
                     [0, 1888, 7016, 12738, 18559, 31937]])

    index=[repeat(a=[0,1,2,3,4],repeats=diff(offsets)[day]) for day in days]    # [0*repeats次, 1*repeats次, ...]

    return index
