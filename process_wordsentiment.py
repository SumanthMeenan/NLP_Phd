#imports
import pandas as pd
# library for plotting graphs
import matplotlib.pyplot as plt
# import from other module fuzzyfy
from fuzzyfy import fuzzyfy_data_by_fixsize

input_folder = "sentiment/"# folder name
#create excel file
def save_dfs_tabs(df_list, sheet_list, file_name):
	"""
	# function for saving multiple dataframes to
	# different tabs of excel sheet

	#parameters:
	# df_list : list of dataframes (tables), contents of table
	# sheet_list : list of column names which will be given to each sheet, name of the excel sheet for each column
	# file_name : name of the excel file to be created, entire file name
	"""
	# open the excel file in write mode with given file name
	writer = pd.ExcelWriter(file_name,engine='xlsxwriter')#in write mode we open... writer is file pointer
	# Loop for each dataframe from list and sheet name from list 
	for dataframe, sheet in zip(df_list, sheet_list):
		#create new sheet in excel file with name sheet_name (max 32 char)
		# start writing data from row = 0 and column=0
		dataframe.to_excel(writer, sheet_name=sheet[:31], startrow=0 , startcol=0)   
	#save the file	
	writer.save()
	print("Excel sheet ", file_name," is saved!")
	return


def fuzzyfy_file(file_name):
	"""
	Function to fuzzyfy given table of data 
	Each column is fuzzified at a time
	Parameters:
	file_name : name of the input data file
	"""
	# read the input data file
	df = pd.read_csv(file_name)
	## Exclude columns containing string / object 
	df = df.select_dtypes(exclude=[object])

	fuzzy_dfs =[]
	for c in df.columns:
		print("current column is ", c)
		# convert data from column to fuzzy format
		# there will be 4 fuzzy levels in data
		fuzzy_df = fuzzyfy_data_by_fixsize(pd.Series(df[c]),4)
		# Transpose the fuzzy data and save it to dataframe (table) format
		fuzzy_df = pd.DataFrame(fuzzy_df.T)
		# append dataframe in list of dataframes
		fuzzy_dfs.append(fuzzy_df)

	#Separate name and extension from given file name
	file_name = file_name.split(".")
	# save all dataframe in excel sheet
	save_dfs_tabs(fuzzy_dfs, df.columns, file_name[0] + '.xlsx')
	print("Fuzzyfied data is saved to xlsx file named " + file_name[0])
	return

def load_and_split_data(file_name):
	"""
	function to split all words in neutral, positive and negative
	"""

	# load word sentiment data from given file
	ws = pd.read_csv(file_name) #dictionary file created in previously
	# if word sentiment value is between -0.000025 to +0.000025 then it is neutral word
	df_neutral = ws[((ws["sentiment_value"] < 0.000025) & (ws["sentiment_value"] > -0.000025))]
	# if word sentiment value is less than -0.000025  then it is negative word
	df_neg = ws[(ws["sentiment_value"] <= -0.000025)]
	# if word sentiment value is greater than +0.000025  then it is positive word
	df_pos = ws[(ws["sentiment_value"] >= 0.000025)]

	print(df_neutral.describe()) #print descriptive statistics df_neutral
	print(df_neg.describe())
	print(df_pos.describe())

	########### Store positive negative and neutral words separately
	df_neutral.to_csv(input_folder+"neutral.csv",index=False) #filder name +column name
	df_neg.to_csv(input_folder+"negative.csv",index=False)
	df_pos.to_csv(input_folder+"positive.csv",index=False)
	print("Word Sentiments saved to files!")
	# return neutral, positive and negative word dataframe
	return df_neutral, df_pos, df_neg

def plot_data(df_neutral, df_pos, df_neg):
	"""
	Function to create histogram plot of positive negative and neutral 
	word sentiment value
	"""
	df_neutral.hist()
	plt.xticks(rotation='vertical')# x axiis label
	plt.legend()#
	plt.show()#


	df_neg.hist()
	plt.xticks(rotation='vertical')
	plt.legend()
	plt.show()

	df_pos.hist()
	plt.xticks(rotation='vertical')
	plt.legend()
	plt.show()

	return 

# load word sentiments and split them in three parts positive negative and neutral 
df_neutral, df_pos, df_neg = load_and_split_data(input_folder+"word_sentiments.csv")
# plot histogram of positive negative and neutral word sentiment values
plot_data(df_neutral, df_pos, df_neg)
# fuzzyfy all neutral word sentiment values 
fuzzyfy_file(input_folder+"neutral.csv")
# fuzzyfy all negative word sentiment values
fuzzyfy_file(input_folder+"negative.csv")
# fuzzyfy all positive word sentiment values
fuzzyfy_file(input_folder+"positive.csv")
print("Completed!")
