import numpy as np 
from utils import sigmoid_prime 


class CostFunction(object) :
	pass
                                                                                                                                                      
class QuadraticCost(CostFunction) :                                                                                                                   
	@staticmethod
	def get_cost(a , y) :
	
		return 0.5 * np.linalg.norm(a - y) ** 2

	@staticmethod
	def get_delta(z , a , y) :

		return (a - y) * sigmoid_prime(z)

class CrossEntropyCost(CostFunction) :

	@staticmethod
	def get_cost(a , y) :
	
		return np.sum(np.nan_to_num(-y*np.log(a) - (1 - y)*np.log(1 - a))) 
	                                                                                                                                                  
	@staticmethod
	def get_delta(z , a , y) :
	
		return (a - y) 
