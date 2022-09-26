import numpy as np 


def init_large_weights(sizes) : 
	
	return [ np.random.randn(x , y) 
			for (x , y) in zip(sizes[1:] , sizes[:-1]) ] 

def init_small_weights(sizes) : 
	
	return [ np.random.randn(x , y) / np.sqrt(x) 
			for (x , y) in zip(sizes[1:] , sizes[:-1]) ] 
