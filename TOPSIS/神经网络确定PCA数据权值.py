#-*-coding:utf-8 -*-
import numpy as np
#首先创建sigmoid函数
def nonlin(x,deriv=False):
    if deriv==True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

#然后创建x和y
x=np.array([[-0.29365471,0.00428678,0],
            [-0.2884454,0.01162937,1],
            [-0.28576096,0.01313499,0],
            [1,0,1]
    ])
y=np.array([[1,0,0,1]]).T

#创建syn0，为权重矩阵。随机生成，注意控制其范围在-1到1
syn0=2*np.random.random([3,1])-1

#训练网络，开始循环。根据差别修改权重矩阵-更新l0层数据-循环n次
for i in range(1000):
    #网络有两层，l0和l1.
    l0=x
    l1=nonlin(np.dot(l0,syn0))
    l1_error=y-l1
    l1_delta=l1_error*nonlin(l1,deriv=True)
    syn0+=np.dot(l0.T,l1_delta)
    
print("the training result is:")
print(l1)
    