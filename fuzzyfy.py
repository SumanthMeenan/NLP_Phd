# imports
import numpy as np
#import library for fuzzy trapezoidal membership function
from skfuzzy.membership import trapmf


def get_range_array_numeric(data):
	"""
	# find mean and standard deviation of numeric data
	# return array of six sigma ranges
	"""
	data_mean = np.mean(data)
	data_std = np.std(data)
	# calculate array of six sigma ranges
	range_array = [	data_mean- 3*data_std,
					data_mean- 2*data_std,
					data_mean- data_std,
					data_mean,
					data_mean+ data_std,
					data_mean+ 2*data_std,
					data_mean+ 3*data_std]
	return range_array


def fuzzyfy_data(data):
	"""
	Apply fuzzy membership function to given data
	"""
	# get six sigma range of data
	range_array = get_range_array_numeric(data)
	fuzzy_data=[]
	# apply fuzzy membership on each 4 set of range values
	# ex. [a= (mean - 3*sigma),b=(mean - 2*sigma),c=(mean - 1*sigma),d=(mean)]
	# store the generated fuzzy data in list / array
	for i in range(len(range_array)-3):
		fuzzy_data.append(trapmf(data,range_array[i:i+4]))
	return fuzzy_data

def fuzzyfy_data_by_fixsize(data,size):
	"""
	Convert given array or list of 'n' elements in fuzzy form
	Size of return array will be 'n' * 'size'
	"""
	data = np.array(data) #convert data into array
	# convert data array in fuzzy form
	fuzzy_data = fuzzyfy_data(data)
	# get size of fuzzy data
	# ex. if it is 2*n then 2 is returned here. 'n' is number of input elements
	s = len(fuzzy_data) #(lenth is no.of rows)
	if(size < s):
		print("Invalid size. Size should be atleast ", s, "for this data")
	else:
		# if size of fuzzy data is less than 'size'(this size is given by capsNet)then append zeros
		# or padding to the data
		# if fuzzy data is 2 * n and given(input) 'size' = 6 then append padding for 
		# 4 size and reshape data to 6 * n   
		p_r = size-s  #size is given by capsNet , s is given by fuzzy function
		fuzzy_data=np.pad(fuzzy_data, [(0,p_r),(0,0)], 'constant', constant_values=(0))
		#0.p_r (0 is first row of above the data and p_r is how many rows are padded below the data) is paddin for the rows, 0,0 padding for the column
	return fuzzy_data

## TRIAL .. 
# sample_arr= np.random.rand(50)
# print(len(fuzzyfy_data_by_fixsize(sample_arr,10)))
# print(fuzzyfy_data_by_fixsize(sample_arr,10))


