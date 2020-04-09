# Tools for feature selection 

import numpy as np
import pandas as pd
# from dcor_jh import distcorr_array
from sklearn.preprocessing import StandardScaler

#------------
# functions for SIS
# could be used to simplify rank_correlation func below
#------------
def xy_correlation_mag(X,y,standardize=True):
    "get correlation magnitude of each column of X with y"
    if standardize==True:
        ss = StandardScaler()
        X = ss.fit_transform(X)
    return np.abs(np.dot(X.T,y))

def top_n_features(X,y,n,standardize=True):
    "get top n features by correlation magnitude"
    corr = xy_correlation_mag(X,y,standardize)
    return np.argsort(corr)[:-n:-1]
	
######

def remove_invariant_cols(df):
	std = df.std()
	zero_var = std.index[std==0]
	print('Removed invariant columns:',zero_var)
	return df.drop(zero_var,axis=1)
	
def remove_invalid_cols(df):
	invalid = df.columns[df.isnull().max()]
	print('Removed columns with invalid values:',invalid)
	return df.drop(invalid,axis=1)
	

def rank_correlation(df,response_col, return_coef=False,corr_type='pearson'):
	"""
	Rank features by correlation to response
	
	Args:
		df: DataFrame with features and response
		response_col: response column name
		return_coef: if True, return ranked coefficients in addition to ranked feature names
		corr_type: correlation coefficient calculation. Options: 'pearson','distance'
	Returns:
		feature_names: feature names sorted by descending correlation with response
		corr
		
	"""
	response_idx = list(df.columns).index(response_col)
	
	if corr_type=='pearson':
		corr = np.corrcoef(df,rowvar=False)
	#elif corr_type=='distance':
	#	corr = distcorr_array(df.values)
	else:
		raise ValueError(f'Invalid correlation type {corr_type}')
	# get magnitude of each feature's correlation to the response
	response_corr = np.abs(np.nan_to_num(corr,0))[response_idx]
	sort_idx = np.argsort(response_corr)[::-1]
	# remove the response column
	sort_idx = sort_idx[sort_idx!=response_idx]
	if return_coef is True:
		return df.columns[sort_idx], response_corr[sort_idx]
	else:
		return df.columns[sort_idx]
		

def get_linked_groups(X, thresh, return_names=True):
	"""
	Get groups of directly and indirectly correlated columns
	
	Args:
		X: matrix or DataFrame. Rows are observations, columns are features
		thresh: correlation coefficient threshold for linking
		return_names: if True and X is DataFrame, return column names. Else return indexes
	"""
	
	corrcoeff = np.corrcoef(X,rowvar=False)
	correlated_columns = np.where(np.abs(np.triu(np.nan_to_num(corrcoeff,0),1))>=thresh)
	
#	  corr_nodiag = corrcoeff - np.diag(np.diag(corrcoeff))
#	  correlated_columns = np.where(np.abs(np.nan_to_num(corr_nodiag,0))>=thresh)
	
	groups = []
	for num in np.unique(correlated_columns[0]):
		in_set = [num in numset for numset in groups]
		if max(in_set,default=False)==False:
			#if not already in a set, get correlated var nums and check if they belong to an existing set
			cnums = correlated_columns[1][np.where(correlated_columns[0]==num)]
			numset = set([num] + list(cnums))
			#check if numset intersects an existing set
			intersect = [numset & group for group in groups]
			if len(intersect) > 0:
				intersect_group = intersect[np.argmax(intersect)]
			else:
				intersect_group = []
			#if intersects existing set, add to set
			if len(intersect_group) > 0:
				intersect_group |= numset
				#print('case 1 existing group:', num, intersect_group)
			#otherwise, make new set
			else:
				groups.append(numset)
				#print('new group:', num, cnums)
		else:
			#if already in a set, get correlated var nums and add to set
			group = groups[in_set.index(True)]
			cnums = correlated_columns[1][np.where(correlated_columns[0]==num)]
			group |= set(cnums) #union
			#print('case 2 existing group:', num, group)
	
	#some links may not be captured. Ex: 1 -> {4,5}. 2 -> 3. 3 -> 4. Now groups are: {1,4,5}, {2,3,4} - need to combine
	#safety net - combine groups that share common elements
	for i,group1 in enumerate(groups):
		for group2 in groups[i+1:]:
			if len(group1 & group2) > 0:
				group1 |= group2
				groups.remove(group2)
				
	if type(X)==pd.core.frame.DataFrame and return_names==True:
		# return column names instead of indexes
		groups = [set([X.columns[idx] for idx in g]) for g in groups]
		
	return groups
	
	
def choose_independent_features(X,thresh,response_col=0,drop_invariant=True):
	"""
	Choose features that correlate best with the response and are not correlated with each other.
	Identify correlation groups and keep the single feature with the strongest correlation to the response from each group.
	
	Args:
		X: matrix or dataframe. Rows are observations, columns are features
		thresh: correlation coefficient threshold
		response_col: column index (or name, if X is a DataFrame) for response. Default 0
		drop_invariant: if True, drop columns with zero variance
	"""
	groups = get_linked_groups(X,thresh=thresh,return_names=False)
	corrcoeff = np.corrcoef(X,rowvar=False)
	#get list of all columns in linked groups
	correlated = sum([list(group) for group in groups],[])
	#get list of unlinked columns 
	independent = set(np.arange(corrcoeff.shape[0])) - set(correlated)
	#for each linked group, keep the feature that correlates most strongly with the response
	if type(response_col)==str:
		# convert column name to index
		response_col = list(X.columns).index(response_col)
	keep = []
	for group in groups:
		max_idx = np.argmax(np.abs(corrcoeff[response_col,list(group)]))
		keep.append(list(group)[max_idx])
	#print(keep)

	keep += list(independent)
	#print(keep)
	
	#check
	check1 = (len(correlated) + len(independent) == corrcoeff.shape[0])
	check2 = (len(correlated) + len(keep) - len(groups) == corrcoeff.shape[0])
	if min(check1,check2)==False:
		raise Exception('Number of correlated and independent features do not match total number')
	
	if drop_invariant==True:
		invariant = list(np.where(np.nan_to_num(corrcoeff,0)[response_col]==0)[0])
		keep = list(set(keep) - set(invariant))
		#print(invariant)
	
	# don't keep the response
	if response_col in keep:
		keep.remove(response_col)
	
	if type(X)==pd.core.frame.DataFrame:
		# return column names instead of indexes
		keep = [X.columns[k] for k in keep]
	
	return keep