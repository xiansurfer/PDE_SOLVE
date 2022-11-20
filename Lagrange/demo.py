# @Time     :2022/9/22 14:47
# @Author   :LuXin
# @Function :实现拉格朗日插值

import numpy as np
import matplotlib.pyplot as plt

start = 0
end = 10
step = .05
fun = lambda x:np.cos(x)

x_samples = np.arange(start,end,step)
y_samples = list(map(lambda x:fun(x)+np.random.uniform(-0.05,0.05),
                     x_samples))

def lagrange_insert(x,x_samples,y_samples):
    lagrange_item = []
    for i in range(len(y_samples)):
        up = y_samples[i]
        down = 1
        for j in range(len(x_samples)):
            if i != j:
                up = up*(x-x_samples[j])
                down = down*(x_samples[i]-x_samples[j])
        lagrange_item.append(up/down)
    return np.sum(lagrange_item)

x_predict = np.arange(start,end,step)
y_predict = []
for x in x_predict:
    y_predict.append(lagrange_insert(x,
                                     x_samples,
                                     y_samples))
plt.plot(x_predict,y_predict)
# plt.scatter(x_samples,y_samples,label='samples')
plt.plot()
plt.show()
print(x_samples,y_samples)
