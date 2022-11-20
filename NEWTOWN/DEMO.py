# @Time     :2022/9/22 18:15
# @Author   :LuXin
# @Function :牛顿插值法

import numpy as np
import matplotlib.pyplot as plt

start = 0
end = 0.5
step = .1

fun = lambda x:np.exp(x)

#生成样本点
x_samples = np.arange(start,end,step)
y_samples = [fun(x)
             for x in x_samples]


def diff_table_cal(x_samples,fun):
    x_dim = len(x_samples)
    diff_table = np.ones((x_dim,x_dim))
    for i in range(x_dim):
        for j in range(x_dim):
            if j == 0:
                diff_table[i][j] = fun(x_samples[i])
            elif j <= i:
                up = diff_table[i-1][j-1] - diff_table[i][j-1]
                down = x_samples[i-j] - x_samples[i]
                diff_table[i][j] = up/down

    return diff_table

def newton_insert(x,diff_table,x_samples):

    n_dim = diff_table.shape[0]
    operator_list = []
    for i in range(n_dim):
        operator = diff_table[i][i]
        if i == 0:
            operator_list.append(operator)
        else:
            k = 0
            while k < i:
                operator *= (x-x_samples[k])

                k += 1

            operator_list.append(operator)
    res = np.sum(operator_list)
    print(operator_list)
    return res


diff_table = diff_table_cal(x_samples=x_samples,
                            fun=fun)
print(diff_table)
y_predict = [newton_insert(x,diff_table,x_samples) for x in x_samples]
plt.plot(x_samples,y_samples,label='smaple')
plt.scatter(x_samples,y_predict,label='newtown')
plt.legend()
plt.show()
# res = newton_insert(test,diff_table,x_samples)









