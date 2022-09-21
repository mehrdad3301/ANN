"""
MNIST loader 
------------ 
 
A library to load MNIST dataset. You can download data at 
http://yann.lecun.com/exdb/mnist/. For details see docstrings 
for ``load_idx`` , `` `` , `` ``.
"""

import numpy as np 
import struct as st 

FILE_ADDRESS = { 
	'train_images' : r'train/train-images-idx3-ubyte' ,
	'train_labels' : r'train/train-labels-idx1-ubyte' , 
	'test_images' : r'test/t10k-images-idx3-ubyte' ,
	'test_labels' : r'test/t10k-labels-idx1-ubyte' ,
}


def load_idx(filename:str) -> np.array : 
	"""Convert data in IDX format to numpy array"""

	file_ = open(filename , 'rb')
	file_.seek(0) 

	magic_number = st.unpack('>4b' , file_.read(4)) 

	dimension_sizes = np.zeros(magic_number[3])  
	for index in range(magic_number[3]) : 
		dimension_sizes[index] = st.unpack('>I' , file_.read(4))[0]

	remaining_bytes = np.prod(dimension_sizes , dtype=np.int)

	data = np.asarray(st.unpack(f'>{remaining_bytes}B' , file_.read(remaining_bytes))).\
	reshape(dimension_sizes.astype(int)) 

	return data  

def load_mnist(validation=False , ratio=0.0) : 
	"""Return a tuple containing train/test data.
	When validation is True, Split training data into train/validation sets
	with given ratio.

	In particular 

	Parameters 
	----------
	validation : bool, optional 
	if True, Split training data into train and validation
	sets and return (train_data , validation_data , test_data).
	
	ratio : float, optional 
	ratio used to split training data into train/validation set.
	""" 

	train_images = load_idx(FILE_ADDRESS['train_images'])
	train_labels = load_idx(FILE_ADDRESS['train_labels'])
	test_images  = load_idx(FILE_ADDRESS['test_images']) 
	test_labels  = load_idx(FILE_ADDRESS['test_labels'])

	train_images = train_images.reshape(-1 , 28 * 28) 
	test_images = test_images.reshape(-1 , 28 * 28) 
	vec_train_labels = [vectorize_labels(x) for x in train_labels] 	
	
	train_data = np.array(list(zip(train_images , vec_train_labels)))
	test_data = np.array(list(zip(test_images , test_labels)))

	if not validation : 
		return (train_data , test_data)
			
	validation_data = np.array(list(zip(train_images[50000:] , train_labels))) 
	train_data = train_data[:50000]

	return (train_data , validation_data , test_data) 

	
def vectorize_labels(j) : 
	"""Converts single letter digits to vector with zeros in all 
	positions except element at position j""" 
	
	x = np.zeros((10 , 1)) 
	x[j] = 1
	return x 	



	
