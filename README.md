# Table of Contents

-[Summary](#summary)

-[Network Structure](#network-structure)

-[Activation Functions](#activation-functions)

-[Prediction](#prediction)

-[Backpropagation](#backpropagation-chain-rule)

-[Training](#gradient-descent-training-of-the-network)

-[References](#references)


## Summary

This repository contains a neural network for MNIST digit recognizer implemented in numpy from scratch.

This is a simple demonstration which shows the basic workflow of a machine learning algorithm using a simple neural network. The derivative at the backpropagation stage is computed explicitly through the chain rule.

* The model is a 3-layer neural network, in which the input layer has 784 units, and the 256-unit hidden layer is activated by ReLU, while the output layer is activated by softmax function to produce a discrete probability distribution for each input. 

* The training is through a standard gradient descent with step size adaptively changing by Root Mean Square prop (RMSprop)


## Network Structure

<img src="https://raw.githubusercontent.com/scaomath/UCI-Math10/master/Lectures/neural_net_3l.png" alt="drawing" width="700"/>

The figure above is a simplication of the neural network used in this example. The circles labeled "+1" are the bias units. Layer 1 is the input layer, and Layer 3 is the output layer. The middle layer, Layer 2, is the hidden layer.

The neural network in the figure above has 2 input units (not counting the bias unit), 3 hidden units, and 1 output unit. In this actual computation below, the input layer has 784 units, the hidden layer has 256 units, and the output layers has 10 units ($K =10$ classes).



## Activation functions
Activation functions play a critical role in neural networks by introducing non-linearity, which enables the network to learn complex patterns in data. Without activation functions, neural networks would act as a linear regression model, failing to capture the real-world complexity of data, in this project I decide to use ReLU and softmax.

* #### ReLU: 
  The Rectified Linear Unit (ReLU) is one of the most popular activation functions used in neural networks, especially in deep learning models. It has become the default choice in many architectures due to its simplicity and efficiency. The ReLU function is a piecewise linear function that outputs the input directly if it is positive; otherwise, it outputs zero.
* #### Softmax: 
  Softmax function is a mathematical function that converts a vector of raw prediction scores (often called logits) from the neural network into probabilities. These probabilities are distributed across different classes such that their sum equals 1. Essentially, Softmax helps in transforming output values into a format that can be interpreted as probabilities, which makes it suitable for classification tasks.


## Prediction

The weight matrix $W^{(0)}$ mapping input $\mathbf{x}$ from the input layer (Layer 1) to the hidden layer (Layer 2) is of shape `(784,256)` together with a `(256,)` bias. Then $\mathbf{a}$ is the activation from the hidden layer (Layer 2) can be written as:
$$
\mathbf{a} = \mathrm{ReLU}\big((W^{(0)})^{\top}\mathbf{x} + \mathbf{b}\big),
$$
where the ReLU activation function is $\mathrm{ReLU}(z) = \max(z,0)$ 

  ```python
  def relu(x):
    x[x<0]=0
    return x
  ```

From the hidden layer (Layer 2) to the output layer (layer 3), the weight matrix $W^{(1)}$ is of shape `(256,10)`, the form of which is as follows:
$$
W^{(1)} =
\begin{pmatrix}
| & | & | & | \\
\boldsymbol{\theta}_1 & \boldsymbol{\theta}_2 & \cdots & \boldsymbol{\theta}_K \\
| & | & | & |
\end{pmatrix},
$$
which maps the activation from Layer 2 to Layer 3 (output layer), and there is no bias because a constant can be freely added to the activation without changing the final output. 

At the last layer, a softmax activation is used, which can be written as follows combining the weights matrix $W^{(1)}$ that maps the activation $\mathbf{a}$ from the hidden layer to output layer:
$$
P\big(y = k \;| \;\mathbf{a}; W^{(1)}\big) = \sigma_k(\mathbf{a}; W^{(1)}) := \frac{\exp\big(\boldsymbol{\theta}^{\top}_k \mathbf{a} \big)}
{\sum_{j=1}^K \exp\big(\boldsymbol{\theta}^{\top}_j \mathbf{a} \big)}.
$$

```python
def softmax(X_in,weights):
    s = np.exp(np.matmul(X_in,weights))
    total = np.sum(s, axis=1).reshape(-1,1)
    return s / total
```

$\{P\big(y = k \;| \;\mathbf{a}; W^{(1)}\big)\}_{k=1}^K$ is the probability distribution of our model, which estimates the probability of the input $\mathbf{x}$'s label $y$ is of class $k$. We denote this distribution by a vector 
$$\boldsymbol{\sigma}:= (\sigma_1,\dots, \sigma_K)^{\top}.$$
We hope that this estimate is as close as possible to the true probability: $1_{\{y=k\}}$, that is $1$ if the sample $\mathbf{x}$ is in the $k$-th class and 0 otherwise. 

Lastly, our prediction $\hat{y}$ for sample $\mathbf{x}$ can be made by choosing the class with the highest probability:
$$
\hat{y} = \operatorname{argmax}_{k=1,\dots,K}  P\big(y = k \;| \;\mathbf{a}; W^{(1)}\big). \tag{$\ast$}
$$

Denote the label of the $i$-th input as $y^{(i)}$, and then the sample-wise loss function is the cross entropy measuring the difference of the distribution of this model function above with the true one $1_{\{y^{(i)}=k\}}$: denote $W = (W^{(0)}, W^{(1)})$, $b = (\mathbf{b})$, let $\mathbf{a}^{(i)}$ be the activation for the $i$-th sample in the hidden layer (Layer 2),
$$
J_i:= J(W,b;\mathbf{x}^{(i)},y^{(i)}) := - \sum_{k=1}^{K} \left\{  1_{\left\{y^{(i)} = k\right\} }
\log P\big(y^{(i)} = k \;| \;\mathbf{a}^{(i)}; W^{(1)}\big)\right\}. \tag{1}
$$

Denote the data sample matrix $X := (\mathbf{x}^{(1)}, \dots, \mathbf{x}^{(N)})^{\top}$, its label vector as $\mathbf{y} := (y^{(1)}, \dots, y^{(N)})$, and then the final loss has an extra $L^2$-regularization term for the weight matrices (not for bias): 
$$
L(W,b; X, \mathbf{y}) := \frac{1}{N}\sum_{i=1}^{N} J_i  + \frac{\alpha}{2} \Big(\|W^{(0)}\|^2 + \|W^{(1)}\|^2\Big),
\tag{2}
$$
where $\alpha>0$ is a hyper-parameter determining the strength of the regularization, the bigger the $\alpha$ is, the smaller the magnitudes of the weights will be after training.

```python
def loss(y_pred,y_true):
    global K 
    K = 10
    N = len(y_true)
    y_true_one_hot_vec = (y_true[:,np.newaxis] == np.arange(K))
    loss_sample = (np.log(y_pred) * y_true_one_hot_vec).sum(axis=1)
    return -np.mean(loss_sample)
```
#### Remark:
* **One Hot Encoding** is a method for converting categorical variables into a binary format. It creates new binary columns (0s and 1s) for each category in the original variable. Each category in the original column is represented as a separate column, where a value of 1 indicates the presence of that category, and 0 indicates its absence.

## Backpropagation (Chain rule)

The derivative of the cross entropy $J$ in (1), for a single sample and its label $(\mathbf{x}, y)$ , with respect to the weights and the bias is computed using the following procedure:
> **Step 1**: Forward pass: computing the activations $\mathbf{a} = (a_1,\dots, a_{n_2})$ from the hidden layer (Layer 2), and $\boldsymbol{\sigma} = (\sigma_1,\dots, \sigma_K)$ from the output layer (Layer 3). 

> **Step 2**: Derivatives for $W^{(1)}$: recall that $W^{(1)} = (\boldsymbol{\theta}_1 ,\cdots,  \boldsymbol{\theta}_K)$ and denote 
$$\mathbf{z}^{(2)} = \big(z^{(2)}_1, \dots, z^{(2)}_K\big)  = (W^{(1)})^{\top}\mathbf{a} =
\big(\boldsymbol{\theta}^{\top}_1 \mathbf{a} ,\cdots,  \boldsymbol{\theta}^{\top}_K \mathbf{a}\big),$$ 
for the $k$-th output unit in the output layer (Layer 3), then
$$
\delta^{(2)}_k
:= \frac{\partial J}{\partial z_k^{(2)}} = \Big\{  P\big(y = k \;| \;\mathbf{a}; W^{(1)}\big)- 1_{\{ y = k\}} \Big\} = \sigma_k - 1_{\{ y = k\}}
$$
and 
$$
\frac{\partial J}{\partial \boldsymbol{\theta}_k}= \frac{\partial J}{\partial z_k^{(2)}}\frac{\partial z_k^{(2)}}{\partial \boldsymbol{\theta}_k} = \delta^{(2)}_k \mathbf{a}.
$$
>
> **Step 3**: Derivatives for $W^{(0)}$, $\mathbf{b}$: recall that $W^{(0)} = (\boldsymbol{w}_1 ,\cdots,  \boldsymbol{w}_{n_2})$, $\mathbf{b} = (b_1,\dots, b_{n_2})$, where $n_2$ is the number of units in the hidden layer (Layer 2), and denote 
$$\mathbf{z}^{(1)} = (z_1^{(1)}, \dots, z_{n_2}^{(1)})  = (W^{(0)})^{\top}\mathbf{x} + \mathbf{b} =
\big(\mathbf{w}^{\top}_1 \mathbf{x} +b_1 ,\cdots,  \mathbf{w}^{\top}_{n_2} \mathbf{x} + b_{n_2}\big),$$ 
for each node $i$ in the hidden layer (Layer $2$), $i=1,\dots, n_2$, then

$$\delta^{(1)}_i : = \frac{\partial J}{\partial z^{(1)}_i}  =
\frac{\partial J}{\partial a_i} 
\frac{\partial a_i}{\partial z^{(1)}_i}=
\frac{\partial J}{\partial \mathbf{z}^{(2)}}
\cdot\left(\frac{\partial \mathbf{z}^{(2)}}{\partial a_i} 
\frac{\partial a_i}{\partial z^{(1)}_i}\right)
\\
=\left( \sum_{k=1}^{K} \frac{\partial J}{\partial {z}^{(2)}_k}
\frac{\partial {z}^{(2)}_k}{\partial a_i}  \right) f'(z^{(1)}_i) = \left( \sum_{k=1}^{K} w_{ki} \delta^{(2)}_k \right) 1_{\{z^{(1)}_i\; > 0\}},
$$
where $1_{\{z^{(1)}_i\; > 0\}}$ is ReLU activation $f$'s (weak) derivative, and the partial derivative of the $k$-th component (before activated by the softmax) in the output layer ${z}^{(2)}_k$ with respect to the $i$-th activation $a_i$ from the hidden layer is the weight $w^{(1)}_{ki}$. Thus
>
$$
\frac{\partial J}{\partial w_{ji}}  = x_j \delta_i^{(1)} ,\;
\frac{\partial J}{\partial b_{i}} = \delta_i^{(1)}, \;\text{ and }\;
\frac{\partial J}{\partial \mathbf{w}_{i}}  = \delta_i^{(1)}\mathbf{x} ,\;
\frac{\partial J}{\partial \mathbf{b}} = \boldsymbol{\delta}^{(1)}.
$$


## Gradient Descent: training of the network

In the training, we use a GD-variant of the RMSprop: for $\mathbf{w}$ which stands for the parameter vector in our model
> Choose $\mathbf{w}_0$, $\eta$, $\gamma$, $\epsilon$, and let $g_{-1} = 1$ <br><br>
>    For $k=0,1,2, \cdots, M$<br><br>
>    &nbsp;&nbsp;&nbsp;&nbsp;  $g_{k} = \gamma g_{k-1} + (1 - \gamma)\, \left|\partial_{\mathbf{w}} L (\mathbf{w}_k)\right|^2$<br><br>
>    &nbsp;&nbsp;&nbsp;&nbsp;    $\displaystyle\mathbf{w}_{k+1} =  \mathbf{w}_k -  \frac{\eta} {\sqrt{g_{k}+ \epsilon}} \partial_{\mathbf{w}} L(\mathbf{w}_k)$  

## References:
* [Stanford Deep Learning tutorial in MATLAB](http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/).
* [Learning PyTorch with examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html).
* [UCI Math 10](https://github.com/scaomath/UCI-Math10). 
* [Automatic Differenciation](https://en.wikipedia.org/wiki/Automatic_differentiation)
* [ReLU](https://www.geeksforgeeks.org/relu-activation-function-in-deep-learning/)
* [Softmax](https://www.geeksforgeeks.org/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/)
* [Chain Rule](https://www.geeksforgeeks.org/chain-rule-derivative-in-machine-learning/)
* [Backpropagation](https://www.geeksforgeeks.org/backpropagation-in-neural-network/)
* [Gradient Descent](https://www.geeksforgeeks.org/gradient-descent-algorithm-and-its-variants/)