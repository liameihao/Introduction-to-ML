# Introduction to ML
##  1. 机器学习是什么

通过一个简单例子（线性模型）来解释
<!-- $$
y=\sum^{n}_{i=1} w_ix+b
$$ --> 

<div align="center"><img style="background: white;" src="https://render.githubusercontent.com/render/math?math=y%3D%5Csum%5E%7Bn%7D_%7Bi%3D1%7D%20w_ix%2Bb"></div>
$y$是预测标签，$w$是特征权重，$x$是特征，$b$是偏差，$n$是特征数量。

单一的模型只是数据拟合（例如最小二乘法），但我们如果有个判断标准，并让机器自动寻找合适的权重值，这便是机器学习。



首先是判断标准，即**Loss**，换句话，就是预测便签和真实标签的距离。

我们这里以**MSE**（均方误差）作为判断标准来衡量这个距离
<!-- $$
Loss=\frac{1}{2n}\sum(y^{pred}-y^{true})^2
$$ --> 

<div align="center"><img style="background: white;" src="https://latex.codecogs.com/svg.latex?Loss%3D%5Cfrac%7B1%7D%7B2n%7D%5Csum(y%5E%7Bpred%7D-y%5E%7Btrue%7D)%5E2"></div>
把之前的式子代入
$$
Loss=\frac{1}{2n}\sum(wx+b-y^{true})^2
$$
我们的任务就是最小化这个距离，这时候就需要偏导数来改变$w$的值（学习过程），进行优化
<!-- $$
\frac{\partial{L}}{\partial{w}}=\frac{1}{n}(wx+b-y^{true})\cdot{w}
$$ --> 

<div align="center"><img style="background: white;" src="https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%7BL%7D%7D%7B%5Cpartial%7Bw%7D%7D%3D%5Cfrac%7B1%7D%7Bn%7D(wx%2Bb-y%5E%7Btrue%7D)%5Ccdot%7Bw%7D"></div>

<!-- $$
\frac{\partial{L}}{\partial{b}}=\frac{1}{n}(wx+b-y^{true})
$$ --> 

<div align="center"><img style="background: white;" src="https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%7BL%7D%7D%7B%5Cpartial%7Bb%7D%7D%3D%5Cfrac%7B1%7D%7Bn%7D(wx%2Bb-y%5E%7Btrue%7D)"></div>

那么对$w$进行更新
<!-- $$
w=w-\frac{\partial{L}}{\partial{w}}
$$ --> 

<div align="center"><img style="background: white;" src="https://latex.codecogs.com/svg.latex?w%3Dw-%5Cfrac%7B%5Cpartial%7BL%7D%7D%7B%5Cpartial%7Bw%7D%7D"></div>

<!-- $$
b=b-\frac{\partial{L}}{\partial{b}}
$$ --> 

<div align="center"><img style="background: white;" src="https://latex.codecogs.com/svg.latex?b%3Db-%5Cfrac%7B%5Cpartial%7BL%7D%7D%7B%5Cpartial%7Bb%7D%7D"></div>

以此类推，直到Loss最小。

