# @Time     :2022/10/6 18:59
# @Author   :LuXin
# @Function :

import numpy as np
import matplotlib.pyplot as plt

fun = lambda x:np.exp(x)

start = 0
end = .5
step = 0.1
x_samples = np.arange(start,end,step)
y_samples = fun(x_samples)

print(x_samples)
print(y_samples)

def diff_table_cal(x_samples,y_samples):
    diff_table = np.ones((len(x_samples),len(x_samples)))
    for i in range(diff_table.shape[0]):
        for j in range(diff_table.shape[0]):
            if j == 0 :
                diff_table[i][j] = y_samples[i]
            elif i >= j:
                up = diff_table[i][j-1]-diff_table[i-1][j-1]
                down = x_samples[i] - x_samples[i-j]
                diff_table[i][j] = up/down
    return diff_table

def newtown_insert(x,diff_table,x_samples):
    operator_list = []
    for i in range(diff_table.shape[0]):
        operator = diff_table[i][i]
        if i == 0:
            operator_list.append(operator)
        else:
            j = 0
            while j < i:
                operator *= (x-x_samples[j])
                j += 1
            operator_list.append(operator)
    print(operator_list)
    res = np.sum(operator_list)

    return res


diff_table = diff_table_cal(x_samples,y_samples)
print(diff_table)
y_predict = [newtown_insert(x,diff_table,x_samples) for x in x_samples]

plt.plot(x_samples,y_samples,label='smples')
plt.scatter(x_samples,y_predict,label='newton')
plt.show()