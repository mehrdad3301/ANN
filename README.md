Breaking 98% barrier on MNIST using only numpy. </br>
The code is heavily based on [neuralnetworksanddeeplearning](https://neuralnetworkanddeeplearning.com) implementation of ANN with minor modifications including :</br>
* Fully matrix-based implementation instead of looping over each example, making it almost 3/2 as fast.
* Using [MNIST](http://yann.lecun.com/exdb/mnist/) in idx format instead of serialized python object. 
* Avoiding overflows in sigmoid, by means of numerical tricks.


