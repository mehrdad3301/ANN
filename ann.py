import numpy as np 
from math import exp 


class Network(object) : 


	def __init__(self , sizes) : 
		
		self.num_layers = len(sizes) 
		self.sizes = sizes 
		self.biases = [ np.random.randn(y , 1) for y in sizes[1:] ]
		self.weights = [ np.random.randn(y , x)
					for x , y in zip(sizes[:-1] , sizes[1:]) ]
		
	
	def feed_forward(self , a) : 

		for w , b in zip(self.weights , self.biases) : 
			a = sigmoid(np.dot( w , a ) + b) 
		return a 

	def SGD(self , train_data , epochs , mini_batch_size , 
			eta , test_data) : 
		
		n = len(train_data) 
		for j in range(epochs) : 
			for mini_batch in gen_mini_batch(
					 train_data , mini_batch_size) :
				self.update_mini_batch(mini_batch , eta) 	
		
			print ("epoch: {0} -> {1} / {2}".format(j ,
			    self.evaluate(test_data) ,len(test_data) )) 

	def update_mini_batch(self , mini_batch , eta) : 
		
		nabla_b = [np.zeros(x.shape) for x in self.biases] 
		nabla_w = [np.zeros(x.shape) for x in self.weights] 

		for x , y in mini_batch : 
			delta_b , delta_w = self.backprop(x , y) 
			nabla_b = [nb + db for db , nb in zip(delta_b,nabla_b)]
			nabla_w = [nw + dw for nw , dw in zip(delta_w,nabla_w)] 

		self.weights = [w - eta/len(mini_batch) * delta 
						for delta , w in zip(nabla_w , self.weights) ] 
		self.biases = [b - eta/len(mini_batch) * delta
						for delta , b in zip(nabla_b , self.biases) ] 
		 
	def backprop(self , x , y) :
				
		nabla_b = [np.zeros(z.shape) for z in self.biases] 
		nabla_w = [np.zeros(z.shape) for z in self.weights] 
	

		#forward pass 
		activation = x 
		activations = [x]
		zs = [] 

		for w , b in zip(self.weights , self.biases) : 
			z = np.dot(w , activation) + b 	
			zs.append(z) 
			activation = sigmoid(z) 
			activations.append(activation) 	
	
		#backward pass 
		delta = (activations[-1] - y) * sigmoid_prime(zs[-1]) 
		nabla_b[-1] = delta 
		nabla_w[-1] = np.dot(delta , activations[-2].T) 

		for l in range(2 , self.num_layers) : 
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
			return (nabla_b, nabla_w)	

		
	def evaluate(self , test_data) : 

		results = [(np.argmax(self.feed_forward(x)) , y )
					for (x , y) in test_data]

		return sum([ int(x == y) for (x , y) in results ] ) 


def gen_mini_batch(data , mini_batch_size) : 

	np.random.shuffle(data)
	for k in range(0 , len(data) , mini_batch_size) : 
		yield data[k:k+mini_batch_size] 
	return 

def sigmoid_(x) : 
	if x > 0 : 
		return 1 / (1 + exp(-x))
	return exp(x) / ( 1 + exp(x) )

sigmoid = np.vectorize(sigmoid_) 
def sigmoid_prime(x) :
	return sigmoid(x) * (1 - sigmoid(x)) 	
