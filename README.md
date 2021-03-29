<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [ ['$','$'], ["\\(","\\)"] ],
      processEscapes: true
    }
  });
</script>
# Introduction to ML
##  1. 机器学习是什么

通过一个简单例子（线性模型）来解释
$$
y=\sum^{n}_{i=1} w_ix+b
$$
$y$是预测标签，$w$是特征权重，$x$是特征，$b$是偏差，$n$是特征数量。

单一的模型只是数据拟合（例如最小二乘法），但我们如果有个判断标准，并让机器自动寻找合适的权重值，这便是机器学习。



首先是判断标准，即**Loss**，换句话，就是预测便签和真实标签的距离。

我们这里以**MSE**（均方误差）作为判断标准来衡量这个距离
$$
Loss=\frac{1}{2n}\sum(y^{pred}-y^{true})^2
$$
把之前的式子代入
$$
Loss=\frac{1}{2n}\sum(wx+b-y^{true})^2
$$
我们的任务就是最小化这个距离，这时候就需要偏导数来改变$w$的值（学习过程），进行优化
$$
\frac{\partial{L}}{\partial{w}}=\frac{1}{n}(wx+b-y^{true})\cdot{w}
$$

$$
\frac{\partial{L}}{\partial{b}}=\frac{1}{n}(wx+b-y^{true})
$$

那么对$w$进行更新
$$
w=w-\frac{\partial{L}}{\partial{w}}
$$

$$
b=b-\frac{\partial{L}}{\partial{b}}
$$

以此类推，直到Loss最小。

