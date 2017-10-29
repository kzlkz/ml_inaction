from numpy import *
import operator
from os import listdir

# 【1】初始化数据  
def init_data():  
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])  
    labels = ['A', 'A', 'B', 'B']  
    return group, labels  
  
  
# 【2】kNN实现  
def classify0(input_set, data_set, labels, k):  
    data_set_size = data_set.shape[0]  
    # 计算距离tile 重复以input_set生成跟data_set一样行数的mat  
    diff_mat = tile(input_set, (data_set_size, 1)) - data_set  
    sq_diff_mat = diff_mat ** 2  
    sq_distances = sq_diff_mat.sum(axis=1)  
    distances = sq_distances ** 0.5  
    # 按照距离递增排序  
    sorted_dist_esc = distances.argsort()  # argsort返回从小到大排序的索引值  
    class_count = {}  # 初始化一个空字典  
    # 选取距离最小的k个点  
    for i in range(k):  
        vote_ilabel = labels[sorted_dist_esc[i]]  
        # 确认前k个点所在类别的出现概率,统计几个类别出现次数  
        class_count[vote_ilabel] = class_count.get(vote_ilabel, 0) + 1  
    # 返回前k个点出现频率最高的类别作为预测分类  
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)  
    return sorted_class_count[0][0]  

if __name__ == "__main__":  
    # 初始化数据  
    dt, lables = init_data()  
    rs = classify0([1, 1], dt, lables, 3)  
    print("result:[%s]" % rs)  