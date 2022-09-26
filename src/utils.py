import numpy as np
from math import exp 
import json 

def shuffle(data) :                                                                                                                              
	"""Returns shuffled data                                                                                                                          
	data is a tuple (x , y) where x is a numpy array                                                                                                  
	containing pixel data, and y is labels. Needless                                                                                                  
	to say these two need to be shuffle together.                                                                                                     
	                                                                                                                                                  
	Note that scipy has better ways of implementing this,                                                                                             
	but i didn't wanna make use of any external libraries                                                                                             
	except for numpy.                                                                                                                                 
	"""                                                                                                                                               
	                                                                                                                                                  
	indices = np.arange(len(data[0]))                                                                                                                 
	np.random.shuffle(indices)                                                                                                                        
	return (data[0][indices] , data[1][indices])                                                                                                      
                                                                                                                                                      
def sigmoid_(x) :                                                                                                                                     
	"""Returns sigmoid function defined by :                                                                                                          
	1 / ( 1 + e^-x ). Note that if x is a large positive                                                                                              
	number exp(x) would overflow. Same thing can happen                                                                                               
	with large negative numbers at exp(-x). Here we used                                                                                              
	the conditional statement to get around that problem                                                                                              
	"""                                                                                                                                               
	if x > 0 :                                                                                                                                        
		return 1 / (1 + exp(-x))                                                                                                                      
	return exp(x) / ( 1 + exp(x) )                                                                                                                    

sigmoid = np.vectorize(sigmoid_) 
                                                                                                                                                      
def sigmoid_prime(x) :                                                                                                                                
	return sigmoid(x) * (1 - sigmoid(x))                                                                                                              

def normalize(data) :                                                                                                                                 
	"""Returns a normalized version data modifying image pixels.
	data is a tuple (x , y) where x is a numpy array containing 
	pixel data and y is a numpy array of the corresponding labels.

	bigger intervals mean biggers activations that can make the 
	neurons saturate easily. Try network without normalization 
	to see the difference. Trust me, It's worth a try.
	"""

	return data[0].astype('float64') / 256 , data[1]                                


def load(filename) : 
	
	f = open(filename , 'r') 
	data = json.load(f) 

	from ann import Network 
	import costFunctions

	cost = getattr(costFunctions , data["cost"]) 
	net = Network(data["sizes"] , cost=cost)
	net.weights = [ np.array(w) for w in data["weights"] ] 
	net.biases = [ np.array(b) for b in data["biases"] ]  
	
	return net 
