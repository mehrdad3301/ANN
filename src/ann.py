import numpy as np 
from utils import sigmoid , sigmoid_prime , shuffle
from costfunctions import CrossEntropyCost


class Network(object) : 


	def __init__(self , sizes , cost=CrossEntropyCost) : 
		
		self.num_layers = len(sizes) 
		self.sizes = sizes 
		self.cost = cost
		self.biases = [ np.random.randn(y , 1) for y in sizes[1:] ]
		self.weights = [ np.random.randn(y , x)
					for x , y in zip(sizes[:-1] , sizes[1:]) ]
		
	
	def feed_forward(self , a) : 
		"""Returns output of network when the input is a"""

		for w , b in zip(self.weights , self.biases) : 
			a = sigmoid(np.matmul( w , a ) + b) 
		return a 

	def SGD(self , train , epochs , mini_batch_size , 
			lambda_=0.0 , 
			eta,
		    test) : 

		"""Runs Stochastic Gradient Descent algorithm on train_data.
		
		parameters 
		-----------------------------------------------------------
		train_data : list of tuples
		a tuple is given in (x , y) where x is data and y is the 
		label assigned to x.	
		
		epochs : integer 
		SGD loops through data epochs times.
		
		mini_batch_size : integer 
	
		eta : floating point 
		learning rate to use when updating weights and biases. 
		
		test_data : list of tuples 
		-----------------------------------------------------------
		""" 
		n = len(train[0]) 
		for j in range(epochs) : 
			for mini_batch in self.gen_mini_batch(train ,
												  mini_batch_size) :

				 self.update_mini_batch(mini_batch , eta , lambda_ , n)
		
			print ("epoch: {0} -> {1} / {2}".format(j ,
			    self.evaluate(test) , n )) 

	def update_mini_batch(self , mini_batch , eta , lambda_ , n) : 
		"""Updates weights and biases. It averages over all 
		training examples in mini_batch""" 
		
		x , y = mini_batch[0] , mini_batch[1]	
		nabla_b , nabla_w = self.backprop(x , y)

		self.weights = [w*(1 - lambda_*eta/n) - eta/len(mini_batch) * delta 
						for delta , w in zip(nabla_w , self.weights) ] 
		self.biases = [b - eta/len(mini_batch) * delta
						for delta , b in zip(nabla_b , self.biases) ] 
		 
	def backprop(self , x , y) :
		"""Returns a tuple (nabla_b , nabla_w) which are derivative 
		of cost function with respect to biases and weight. nabla_b 
		and nabla_w have the same shapes as biases and weights and 
		can be used to update these values."""
		
				
		nabla_b = [None for _ in self.biases]
		nabla_w = [None for _ in self.weights] 
	
		activation = x 
		activations = [activation]
		zs = [] 

		#forward pass 
		for w , b in zip(self.weights , self.biases) : 
			z = np.matmul(w , activation) + b 	
			zs.append(z) 
			activation = sigmoid(z) 
			activations.append(activation) 	
	
		delta = self.cost.delta(zs[-1] , activations[-1] , y)
		nabla_b[-1] = np.sum(delta , axis=0)  
		nabla_w[-1] = np.sum(np.matmul(delta ,
		 activations[-2].transpose(0 ,2 , 1)),axis=0) 

		#backward pass 
		for l in range(2 , self.num_layers) : 
			z = zs[-l]
			sp = sigmoid_prime(z)
			delta = np.matmul(self.weights[-l+1].transpose(), delta) * sp
			nabla_b[-l] = np.sum(delta , axis = 0)
			nabla_w[-l] = np.sum(np.matmul(delta, activations[-l-1].transpose(0 , 2 ,1)),	axis=0)
			return (nabla_b, nabla_w)	

		
	def evaluate(self , test) : 
		"""Returns the number of correct outputs in test_data"""

		return np.sum(
			np.argmax( self.feed_forward(test[0]) , axis=1 ) == test[1].reshape(-1 , 1) 
		)


	def gen_mini_batch(self , data , mini_batch_size) : 
		"""Returns a generator, yielding a mini-batch with 
		given size on each time it is evoked """ 
	
		data = shuffle(data) 
		for k in range(0 , len(data[0]) , mini_batch_size) : 
			yield (data[0][k:k+mini_batch_size] ,
					data[1][k:k+mini_batch_size] ) 
		return 



