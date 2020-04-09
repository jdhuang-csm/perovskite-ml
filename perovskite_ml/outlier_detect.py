# Module for outlier detection and removal 

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from .plotting import scatter_slices, add_colorbar
from .quaternary_plt import QuaternaryAxes
import pymatgen as mg
import matplotlib as mpl

def z_score(x):
	mu = np.mean(x)
	std = np.std(x)
	return (x-mu)/std

class DataCleaner():
	"""
	Class for outlier detection and data cleaning in preprocessing
	Implements sklearn.cluster.DBSCAN for compositional clustering, 
	and sklearn.ensemble.IsolationForest for greedy outlier flagging.
	Applies z-score threshold within composition clusters to screen 
	IsolationForest flags.
	
	Parameters
	----------
	data: dataset to process (pandas DataFrame)
	prop_dim: property dimension to screen for outliers
	comp_dims: composition dimensions for clustering and IsolationForest
	add_fit_dims: additional dimensions to use for outlier identification
	cluster_by: column to group by for clustering. If None, use DBSCAN to identify clusters
	DB_kw: kwargs to pass to DBSCAN instantiation
	IF_kw: kwargs to pass to IsolationForest instantiation
	"""
	def __init__(self, data, prop_dim, comp_dims=None, add_fit_dims=[],cluster_by=None, DB_kw={},IF_kw={}):
		self.data = data
		self.set_prop_dim(prop_dim)
		self.set_comp_dims(comp_dims)
		self.add_fit_dims = add_fit_dims
		self.cluster_by = cluster_by
		self.random_state = np.random.RandomState(17)
		self.db = DBSCAN(**DB_kw)
		self.clf = IsolationForest(random_state=self.random_state,**IF_kw)
				 
	def set_prop_dim(self,prop_dim):
		"set property dimension"
		self._prop_dim = prop_dim
	
	def get_prop_dim(self):
		"get property dimension"
		return self._prop_dim
	
	prop_dim = property(get_prop_dim,set_prop_dim)
	
	def set_comp_dims(self,comp_dims=None):
		"""
		set composition dimensions used for clustering. Defaults to all valid elemental symbols in data columns
		
		Parameters
		----------
		comp_dims: list of columns in data to use as composition dimensions
		"""
		#if no comp dims specified, use all columns that are valid element symbols
		if comp_dims==None:
			comp_dims = []
			for col in self.data.columns:
				try: 
					mg.Element(col)
					comp_dims.append(col)
				except ValueError:
					pass
		self._comp_dims = comp_dims
		
	def get_comp_dims(self):
		"get composition dimensions"
		return self._comp_dims
	
	comp_dims = property(get_comp_dims,set_comp_dims)
	
	@property 
	def comp_data(self):
		"composition data"
		return self.data[self.comp_dims]
	
	@property
	def fit_dims(self):
		"dimensions used for identifying outliers"
		return self.comp_dims + self.add_fit_dims + [self.prop_dim]
	
	def fit_data(self,comp_scale=1,prop_scale=1,add_fit_scale=1,cluster_by=None):
		"data used for identifying outliers"
		fit_data = self.data.copy()
		fit_data[self.comp_dims] = self.scaled_comp_data(scale=comp_scale,cluster_by=cluster_by).values
		
		ss = StandardScaler()
		if cluster_by is None:
			fit_data[self.prop_dim] = prop_scale*ss.fit_transform(fit_data[self.prop_dim].values[:,None])
			if len(self.add_fit_dims)>0:
				fit_data[self.add_fit_dims] = add_fit_scale*ss.fit_transform(fit_data[self.add_fit_dims])
		else:
			# scale within each cluster
			gdf = fit_data.groupby(cluster_by)
			for cluster, idx in gdf.groups.items():
				cdata = fit_data.loc[idx,:]
				fit_data.loc[idx,self.prop_dim] = prop_scale*ss.fit_transform(cdata[self.prop_dim].values[:,None])
				if len(self.add_fit_dims)>0:
					fit_data.loc[idx,self.add_fit_dims] = add_fit_scale*ss.fit_transform(cdata[self.add_fit_dims])
		
		return fit_data[self.fit_dims]
	
	def scaled_comp_data(self,scale=1,cluster_by=None):
		"""
		scale composition dimensions such that largest-variance dimension has variance max_var
		"""
		ss = StandardScaler()
		if cluster_by is None:
			#get dimension with largest variance
			ref_dim = np.var(self.comp_data).idxmax()
			ss.fit(self.comp_data[ref_dim].values[:,None])
			#scale all comp dims with same scaler such that refdim has variance max_var
			scaled_comp_data = pd.DataFrame(scale*ss.transform(self.comp_data),columns=self.comp_dims)
		else:
			# scale within each cluster
			gdf = self.data.groupby(cluster_by)
			scaled_comp_data = self.comp_data.copy()
			for cluster, idx in gdf.groups.items():
				cdata = self.comp_data.loc[idx,:]
				#get dimension with largest variance
				ref_dim = np.var(cdata).idxmax()
				ss.fit(cdata[ref_dim].values[:,None])
				#scale all comp dims with same scaler such that refdim has variance max_var
				scaled_comp_data.loc[idx,:] = scale*ss.transform(cdata)
				
		return scaled_comp_data
				
				
		
	def fit(self, method,comp_scale=1,prop_scale=1):
		"""
		fit DBSCAN and IsolationForest to data
		
		Parameters
		----------
		comp_scale: maximum compositional variance set by scale_composition
		"""
		
		if method=='DBIFZ':		
			
			if self.cluster_by is None:
				# fit DBSCAN to comp data for compositional clustering
				self.db.fit(self.scaled_comp_data(scale=comp_scale)) 
				
			# fit IsolationForest to iso data for greedy outlier flagging
			self.clf.fit(self.fit_data(comp_scale,prop_scale)) 
			
		elif method=='DBSCAN':
			# nothing to do yet
			pass
			
			
	def predict(self,method,comp_scale=1,prop_scale=1,add_fit_scale=1,z_thresh=2):
		"""
		predict outliers in data
		
		Parameters
		----------
		z_thresh: z-score threshold for intra-cluster outlier identification
		"""
		
		self.pred = pd.DataFrame()
		self.pred[self.prop_dim] = self.data[self.prop_dim]
		self.z_thresh = z_thresh
		
		if self.cluster_by is not None:
			# use provided column to group into clusters
			self.pred.loc[:,'cluster_name'] = self.data[self.cluster_by]
			cluster_names = self.pred['cluster_name'].unique()
			clusters = np.arange(len(cluster_names))
			cluster_dict = dict(zip(cluster_names,clusters))
			self.pred['cluster'] = self.pred['cluster_name'].map(lambda x: cluster_dict[x])
			self.cluster_name = dict(zip(clusters,cluster_names))
		
		if method=='DBIFZ':
			
			if self.cluster_by is None:
				# use DBSCAN to cluster by composition
				self.pred.loc[:,'cluster'] = self.db.fit_predict(self.scaled_comp_data(comp_scale)) #db has no pure predict function
				self.pred['cluster_name'] = self.pred['cluster']
				clusters = self.pred['cluster'].unique()
				self.cluster_name = dict(zip(clusters,clusters))
				
			fit_data = self.fit_data(comp_scale,prop_scale,add_fit_scale)
			
			self.pred.loc[:,'isolation_flag'] = self.clf.predict(fit_data)
			self.pred.loc[:,'isolation_score'] = self.clf.decision_function(fit_data)

			#get z-scores for each cluster and cross-ref with isolation forest
			for i, cluster in enumerate(self.pred['cluster'].unique()):
				df = self.pred.loc[self.pred['cluster']==cluster,:]
				self.pred.loc[self.pred['cluster']==cluster,'cluster_zscore'] = z_score(df[self.prop_dim])

			#set final outlier flag - if flagged by isolation forest and cluster z-score is outside z_thresh
			self.pred.loc[:,'outlier_flag'] = np.where(
												(self.pred['isolation_flag']==-1) & (np.abs(self.pred['cluster_zscore']) > z_thresh),
												-1, 0)
			
		elif method=='DBSCAN':
			
			if self.cluster_by is None:
				# apply DBSCAN to comp dims and prop dim to cluster and identify outliers
				fit_data = self.fit_data(comp_scale,prop_scale,add_fit_scale)
				self.pred.loc[:,'cluster'] = self.db.fit_predict(fit_data) #db has no pure predict function
				self.pred['cluster_name'] = self.pred['cluster']
				clusters = self.pred['cluster'].unique()
				self.cluster_name = dict(zip(clusters,clusters))
				# cluster -1 is outliers
				self.pred['outlier_flag'] = self.pred['cluster'].map(lambda x: -1 if x==-1 else 0)
				
			else:
				# apply DBSCAN within each provided cluster to identify outliers
				fit_data = self.fit_data(comp_scale,prop_scale,add_fit_scale,cluster_by=self.cluster_by)
				for cluster, idx in self.data.groupby(self.cluster_by).groups.items():
					cdata = fit_data.loc[idx,:]
					self.pred.loc[idx,'DB_cluster'] = self.db.fit_predict(cdata)
				# cluster -1 is outliers
				self.pred['outlier_flag'] = self.pred['DB_cluster'].map(lambda x: -1 if x==-1 else 0)
				
			#get z-scores for each cluster
			for i, cluster in enumerate(self.pred['cluster'].unique()):
				df = self.pred.loc[self.pred['cluster']==cluster,:]
				self.pred.loc[self.pred['cluster']==cluster,'cluster_zscore'] = z_score(df[self.prop_dim])
			
			# set IF columns for compatibility
			self.pred.loc[:,'isolation_flag'] = 1
			self.pred.loc[:,'isolation_score'] = 0
				
		#include scaled fit_data in pred
		for col in fit_data.columns:
			self.pred[f'{col}_fit'] = fit_data[col]
		#return self.pred
		
	def fit_predict(self,method,comp_scale=1,prop_scale=1,add_fit_scale=1,z_thresh=2):
		"""combine fit and predict functions"""
		self.fit(method,comp_scale,prop_scale)
		self.predict(method,comp_scale,prop_scale,add_fit_scale,z_thresh)
		#return self.pred
	
	def remove_outliers(self):
		"""remove outliers identified by fit_predict"""
		self.clean_data = self.data[self.pred['outlier_flag']!=-1]
		#return self.clean_data
	
	@property
	def data_pred(self):
		"data joined with prediction results"
		return self.data.join(self.pred.drop(labels=self.prop_dim, axis=1))
	
	@property
	def outliers(self):
		"outlier data rows"
		return self.data_pred[self.data_pred['outlier_flag']==-1]
	
	@property
	def inliers(self):
		"inlier data rows"
		return self.data_pred[self.data_pred['outlier_flag']!=-1]
	
	def set_DB_params(self,**params):
		"""set DBSCAN parameters"""
		self.db.set_params(**params)
	
	def set_IF_params(self,**params):
		"""set IsolationForest parameters"""
		self.clf.set_params(**params)
		
	def scatter_slices(self, slice_axis, slice_starts, slice_widths, tern_axes,color_col=None,vmin=None,vmax=None,cmap=plt.cm.viridis,data_filter=None,**scatter_kwargs):
		if color_col is None:
			color_col = self.prop_dim
		
		if data_filter is not None:
			data = data_filter(self.data_pred)
		else:
			data = self.data_pred
		
		#get vmin and vmax
		if vmin is None:
			vmin = data[color_col].min()
		if vmax is None:
			vmax = data[color_col].max()
		
		#plot all
		axes = scatter_slices(data,color_col,slice_axis,slice_starts,slice_widths,tern_axes,cmap=cmap,
							vmin=vmin,vmax=vmax,**scatter_kwargs)
							
	def scatter_slice_highlight(self, slice_axis, slice_starts, slice_widths, tern_axes,color_col=None,vmin=None,vmax=None,cmap=plt.cm.viridis,data_filter=None,**scatter_kwargs):
		"""
		plot all data points with outliers highlighted in red. color determined by value of prop_dim
		
		Parameters
		----------
		
		slice_axis: composition dimension on which to slice
		slice_starts: values of slice_axis at which to start slices
		slice_widths: widths of slices in slice_axis dimension. Single value or list
		tern_axes: composition dimensions for ternary plot axes (order: right, top, left)
		cmap: colormap for prop_dim values
		scatter_kwargs: kwargs to pass to helpers.plotting.scatter_slices
		"""
		if color_col is None:
			color_col = self.prop_dim
		
		if data_filter is not None:
			data = data_filter(self.data_pred)
		else:
			data = self.data_pred
		
		#get vmin and vmax
		if vmin is None:
			vmin = data[color_col].min()
		if vmax is None:
			vmax = data[color_col].max()
			
		inliers = data[data['outlier_flag']==0]
		outliers = data[data['outlier_flag']==-1]
			
		#plot inliers
		axes = scatter_slices(inliers,color_col,slice_axis,slice_starts,slice_widths,tern_axes,cmap=cmap,
							vmin=vmin,vmax=vmax,**scatter_kwargs)
		#plot outliers
		scatter_slices(outliers,color_col,slice_axis,slice_starts,slice_widths,tern_axes,cmap=cmap,axes=axes,
					   vmin=vmin,vmax=vmax, colorbar=False, s=20,marker='d',edgecolors='r',linewidths=0.8,**scatter_kwargs)
		
	def scatter_slice_clusters(self, slice_axis, slice_starts, slice_widths, tern_axes, cmap=plt.cm.plasma,**scatter_kwargs):
		"""
		plot all data points with cluster shown by color
		
		Parameters
		----------
		slice_axis: composition dimension on which to slice
		slice_starts: values of slice_axis at which to start slices
		slice_widths: widths of slices in slice_axis dimension. Single value or list
		tern_axes: composition dimensions for ternary plot axes (order: right, top, left)
		cmap: colormap for cluster values
		scatter_kwargs: kwargs to pass to helpers.plotting.scatter_slices
		"""
		#make norm for discrete colormap
		clusters = list(self.cluster_name.keys())
		cluster_names = list(self.cluster_name.values())
		n_clusters = len(self.pred['cluster'].unique())
		bounds = np.arange(min(clusters)-0.5,max(clusters)+0.51)
		norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
		
		scatter_slices(self.data_pred,'cluster',slice_axis,slice_starts,slice_widths,tern_axes,cmap=cmap,norm=norm,
					cb_kwargs={'norm':norm,'ticks':clusters,'tickformat':'%.0f','ticklabels':cluster_names},**scatter_kwargs)
		
	def scatter_slice_outliers(self, slice_axis, slice_starts, slice_widths, tern_axes, cmap=plt.cm.viridis,**scatter_kwargs):
		"""
		plot outliers only
		
		Parameters
		----------
		slice_axis: composition dimension on which to slice
		slice_starts: values of slice_axis at which to start slices
		slice_widths: widths of slices in slice_axis dimension. Single value or list
		tern_axes: composition dimensions for ternary plot axes (order: right, top, left)
		cmap: colormap for prop_dim values
		scatter_kwargs: kwargs to pass to helpers.plotting.scatter_slices
		"""
		axes = scatter_slices(self.outliers,self.prop_dim,slice_axis,slice_starts,slice_widths,tern_axes,cmap=cmap,**scatter_kwargs)
		return axes
		
	def scatter_slice_inliers(self, slice_axis, slice_starts, slice_widths, tern_axes, cmap=plt.cm.viridis,**scatter_kwargs):
		"""
		plot inliers only
		
		Parameters
		----------
		slice_axis: composition dimension on which to slice
		slice_starts: values of slice_axis at which to start slices
		slice_widths: widths of slices in slice_axis dimension. Single value or list
		tern_axes: composition dimensions for ternary plot axes (order: right, top, left)
		cmap: colormap for prop_dim values
		scatter_kwargs: kwargs to pass to helpers.plotting.scatter_slices
		"""
		axes = scatter_slices(self.inliers,self.prop_dim,slice_axis,slice_starts,slice_widths,tern_axes,cmap=cmap,**scatter_kwargs)
		return axes
	
	def cluster_hist(self,ncols=2,cluster_by=None):
		
		if cluster_by is None:
			clusters = list(self.cluster_name.keys())
		else:
			gdf = self.data_pred.groupby(cluster_by)
			clusters = [k for k in gdf.groups.keys()]
			
		#print(clusters)
		nrows = int(np.ceil(len(clusters)/ncols))
		#print(nrows)
		fig, axes = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*3))
		for (i, cluster),ax in zip(enumerate(clusters),axes.ravel()):
			if cluster_by is None:
				df = self.data_pred.loc[self.data_pred['cluster']==cluster,:]
			else:
				idx = gdf.groups[cluster]
				df = self.data_pred.loc[idx,:]

			num_outliers = len(df[df['isolation_flag']==-1])
			# try: #2d axes
				# ax = axes[int(i/ncols), i%ncols]
			# except IndexError: #1d axes
				# ax = axes[i]
				
			dfo = df[df['isolation_flag']==-1]
			dfi = df[df['isolation_flag']==1]
			hist, bins = np.histogram(df['cluster_zscore'])
			if len(dfo) > 0:
				# if isolation forest outliers exist (method=DBIFZ)
				ax.hist([dfo['cluster_zscore'],dfi['cluster_zscore']],alpha=0.8,bins=bins,histtype='barstacked',label=['IsolationForest outliers','IsolationForest inliers'],color=['#ff7f0e','#1f77b4'])
				ax.legend()
			else:
				# if no isolation forest outliers (method=DBSCAN)
				dfo = df[df['outlier_flag']==-1]
				dfi = df[df['outlier_flag']==0]
				ax.hist([dfo['cluster_zscore'],dfi['cluster_zscore']],alpha=0.8,bins=bins,histtype='barstacked',label=['Outliers','Inliers'],color=['#ff7f0e','#1f77b4'])
				ax.legend()
			
			if cluster_by is None:
				ax.set_title('Cluster {}'.format(self.cluster_name[cluster]))
			else:
				ax.set_title('Cluster {}'.format(cluster))
			ax.set_xlabel('Cluster Z-score')
			ax.set_ylabel('Frequency')
			
			#plot z-score threshold
			ax.axvline(-self.z_thresh,ls='--',c='r')
			ax.axvline(self.z_thresh,ls='--',c='r')
			
			# add second axis to show prop_dim values
			ax2 = ax.twiny()
			ax2.set_xlim(ax.get_xlim())
			ax2.set_xticks(ax.get_xticks())
			tick_vals = df[self.prop_dim].mean() + df[self.prop_dim].std()*ax.get_xticks()
			ax2.set_xticklabels(np.round(tick_vals,1))
			ax2.set_xlabel(self.prop_dim)
		fig.tight_layout()
		
	def cluster_scatter(self,x_col,y_col,plot_combined=False,cluster_by=None,flag_outliers=False,ncols=2,s=8,data_filter=None,sharex=False, sharey=False,basefontsize=11,**scatter_kw):
		"""
		Scatter plot for each cluster. 
		
		Args:
			x_col: x column
			y_col: y column
			plot_combined: if True, create an additional plot with all samples overlaid
			cluster_by: column to use for grouping. If None, use clusters assigned by fit_predict
			flag_outliers: if True, plot outliers in orange
			ncols: number of columns for subplot grid
			s: point size
			data_filter: function to filter data. Should apply to DataFrame and return filtered DataFrame.
				Ex: data_filter = lambda df: df[df['property']==value]
			sharex, sharey: kwargs for plt.subplots()
			scatter_kw: kw for plt.scatter()
		"""
		
		if data_filter is None:
			data = self.data_pred
		else:
			data = data_filter(self.data_pred)
		
		if cluster_by is None:
			clusters = list(self.cluster_name.keys())
		else:
			gdf = data.groupby(cluster_by)
			clusters = [k for k in gdf.groups.keys()]	
			
		if plot_combined:
			num_plots = len(clusters)+1
		else:
			num_plots = len(clusters)
		nrows = int(np.ceil(num_plots/ncols))
			
		fig, axes = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*3),sharex=sharex,sharey=sharey)
		
		for (i, cluster),ax in zip(enumerate(clusters),axes.ravel()):
			if cluster_by is None:
				df = data.loc[self.data_pred['cluster']==cluster,:]
			else:
				idx = gdf.groups[cluster]
				df = data.loc[idx,:]
				
			if flag_outliers is False:
				ax.scatter(df[x_col],df[y_col],s=s,**scatter_kw)
			else:
				dfi = df[df['outlier_flag']==0]
				dfo = df[df['outlier_flag']==-1]
				ax.scatter(dfi[x_col],dfi[y_col],label='Inliers',s=s,**scatter_kw)
				ax.scatter(dfo[x_col],dfo[y_col],label='Outliers',s=s,**scatter_kw)
				ax.legend(fontsize=basefontsize)
				
			if cluster_by is None:
				ax.set_title('Cluster {}'.format(self.cluster_name[cluster]),fontsize=basefontsize+1)
			else:
				ax.set_title('Cluster {}'.format(cluster),fontsize=basefontsize+1)
			ax.set_xlabel(x_col,fontsize = basefontsize)
			ax.set_ylabel(y_col,fontsize = basefontsize)
			ax.tick_params(axis='both', which='major', labelsize=basefontsize-1)
			
		if plot_combined:
			# plot all clusters on same axes
			if flag_outliers is False:
				axes.ravel()[-1].scatter(data[x_col],data[y_col],s=s,**scatter_kw)
			else:
				dfi = data[data['outlier_flag']==0]
				dfo = data[data['outlier_flag']==-1]
				axes.ravel()[-1].scatter(dfi[x_col],dfi[y_col],label='Inliers',s=s,**scatter_kw)
				axes.ravel()[-1].scatter(dfo[x_col],dfo[y_col],label='Outliers',s=s,**scatter_kw)
				axes.ravel()[-1].legend(fontsize=basefontsize)
				
			axes.ravel()[-1].set_xlabel(x_col,fontsize = basefontsize)
			axes.ravel()[-1].set_ylabel(y_col,fontsize = basefontsize)
			axes.ravel()[-1].set_title('All Clusters',fontsize = basefontsize +1)
			axes.ravel()[-1].tick_params(axis='both', which='major', labelsize=basefontsize-1)
			
		for ax in axes.ravel()[num_plots:]:
			# turn off unused axes
			ax.axis('off')
			
		if sharex:
			for ax in axes[:-1,:]:
				ax.set_xlabel('')
		if sharey:
			for ax in axes[:,1:].ravel():
				ax.set_ylabel('')
			
		fig.tight_layout()		
		
	def quat_plot(self,ax=None,figsize=(8,6),quat_axes=['Co','Fe','Zr','Y'],label_kw={}, gridlines=True, color_col=None,colorbar=True,cb_kw={}, s=3,data_filter=None,**scatter_kw):
		qax = QuaternaryAxes(ax=ax,figsize=figsize)
		qax.draw_axes()
		# default corner label kwargs
		label_kwargs = dict(offset=0.11,size=14)
		# update with user kwargs
		label_kwargs.update(label_kw)
		qax.label_corners(quat_axes,**label_kwargs)
		
		if color_col is None:
			color_col = self.prop_dim
			
		# Default colorbar kwargs
		cb_kwargs={'label':color_col,'cbrect':[0.8,0.1,0.02,0.65],'labelkwargs':{'size':14},'tickparams':{'labelsize':13}}
		# update with any user-specified kwargs
		cb_kwargs.update(cb_kw)
				
		if data_filter is not None:
			data = data_filter(self.data_pred)
		else:
			data = self.data_pred
				
		if 'vmin' not in scatter_kw.keys():
			scatter_kw['vmin'] = data[color_col].min()
		if 'vmax' not in scatter_kw.keys():
			scatter_kw['vmax'] = data[color_col].max()
		
		qax.scatter(data[quat_axes].values,c=data[color_col], s=s, colorbar=colorbar,cb_kwargs=cb_kwargs,**scatter_kw)
		
		qax.axes_ticks(size=13,corners='rbt',offset=0.08)
		if gridlines==True:
			qax.gridlines(ls=':',LW=0.6)
		qax.ax.axis('off')
	
		return qax
	
	def quat_highlight(self,ax=None,figsize=(8,6),quat_axes=['Co','Fe','Zr','Y'],label_kw={}, gridlines=True,color_col=None,cb_label=None,data_filter=None, **scatter_kw):
		qax = QuaternaryAxes(ax=ax,figsize=figsize)
		qax.draw_axes()
		# default corner label kwargs
		label_kwargs = dict(offset=0.11,size=14)
		# update with user kwargs
		label_kwargs.update(label_kw)
		qax.label_corners(quat_axes,**label_kwargs)
		
		if color_col is None:
			color_col = self.prop_dim
		if cb_label is None:
			cb_label = color_col
			
		if data_filter is not None:
			data = data_filter(self.data_pred)
		else:
			data = self.data_pred
			
		if 'vmin' not in scatter_kw.keys():
			scatter_kw['vmin'] = data[color_col].min()
		if 'vmax' not in scatter_kw.keys():
			scatter_kw['vmax'] = data[color_col].max()
			
		inliers = data[data['outlier_flag']==0]
		outliers = data[data['outlier_flag']==-1]
		
		qax.scatter(inliers[quat_axes].values,c=inliers[color_col],s=3,colorbar=True, 
					cb_kwargs={'label':cb_label,'cbrect':[0.8,0.1,0.02,0.65],'labelkwargs':{'size':14},'tickparams':{'labelsize':13}}, **scatter_kw)
		qax.scatter(outliers[quat_axes].values,c=outliers[color_col],s=6, 
					edgecolors='r',linewidths=0.5, **scatter_kw)
		
		qax.axes_ticks()
		if gridlines==True:
			qax.gridlines()
		qax.ax.axis('off')
	
		return qax
	
	def quat_clusters(self,ax=None,figsize=(8,6),quat_axes=['Co','Fe','Zr','Y'],label_kw={}, gridlines=True, cmap=plt.cm.plasma,s=3, colorbar=True,cb_kw={},
					  **scatter_kw):
		qax = QuaternaryAxes(ax=ax,figsize=figsize)
		qax.draw_axes()
		# default corner label kwargs
		label_kwargs = dict(offset=0.11,size=14)
		# update with user kwargs
		label_kwargs.update(label_kw)
		qax.label_corners(quat_axes,**label_kwargs)
		
		vmin = self.pred['cluster'].min()
		vmax = self.pred['cluster'].max()
		
		#make norm for discrete colormap
		clusters = list(self.cluster_name.keys())#pred['cluster'].unique().astype(int)
		cluster_names = list(self.cluster_name.values())
		n_clusters = len(clusters)
		bounds = np.arange(min(clusters)-0.5,max(clusters)+0.51)
		norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
			
		# Default colorbar kwargs
		if self.cluster_by is None:
				cb_label = 'Cluster'
		else:
			cb_label = self.cluster_by
		cb_kwargs={'label':cb_label,'norm':norm,'ticks':clusters,'ticklabels':cluster_names,
					'cbrect':[0.8,0.1,0.02,0.65],'labelkwargs':{'size':14},'tickparams':{'labelsize':13}}
		# update with any user-specified kwargs
		cb_kwargs.update(cb_kw)
		
		qax.scatter(self.data[quat_axes].values,c=self.pred['cluster'],s=s, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, 
					colorbar=colorbar, cb_kwargs=cb_kwargs, **scatter_kw)
		
		qax.axes_ticks(size=13,corners='rbt',offset=0.08)
		if gridlines==True:
			qax.gridlines(ls=':',LW=0.6)
		qax.ax.axis('off')
		return qax
	
	def reduce_comp_dims(self,kernel='poly',gamma=10,**kpca_kw):
		comp_dims = self.comp_dims.copy()
		if 'O' in comp_dims:
			comp_dims.remove('O')
		if 'Ba' in comp_dims:
			comp_dims.remove('Ba')
		print('Dimensions for KPCA reduction:', comp_dims)
		self.kpca_dims = comp_dims
		#self.reconstructed = self.data.copy()
		# reconstructed dims
		rc_dims = [f'{d}_kpca' for d in comp_dims]
		self.kpca = KernelPCA(kernel=kernel,n_components=2,fit_inverse_transform=True,gamma=gamma,**kpca_kw)

		# self.reduced = self.data_pred.copy()
		# write reduced dimensions to pred (can't write to data_pred - it is basically just a SQL view)
		self.pred['v1'] = 0
		self.pred['v2'] = 0
		self.pred[['v1','v2']] = self.kpca.fit_transform(self.data[comp_dims])
		self.pred[rc_dims] = pd.DataFrame(self.kpca.inverse_transform(self.pred[['v1','v2']]),index=self.pred.index)

		# self.reduced[self.prop_dim] = self.data[self.prop_dim].values
		# self.reduced['outlier_flag'] = self.pred['outlier_flag'].values
		# self.reduced['cluster'] = self.pred['cluster'].values
		
		error = np.linalg.norm(self.data[comp_dims].values - self.pred[rc_dims].values,ord=2)
		print('Reconstruction error:',error)
		
		#return self.reduced, error
		
	def quat_reconstruction(self,ax=None,figsize=(8,6), gridlines=True, color_col=None,cb_label=None, s=3,data_filter=None,**scatter_kw):
		"""
		Plot reconstructed composition data
		"""
		rc_dims = [f'{d}_kpca' for d in self.kpca_dims]
		self.quat_plot(ax,figsize,rc_dims, gridlines, color_col,cb_label, s,data_filter,**scatter_kw)
	
	
	def reduced_plot(self,ax=None,cmap=plt.cm.viridis,vmin=None,vmax=None,cbar=True, cbrect=[0.88,0.12,0.02,0.75],**kwargs):
		"""
		scatter plot of prop_dim in reduced-dimension composition space
		
		Args:
		-----
		kwargs: kwargs to pass to plt.scatter
		"""
		if ax is None:
			fig, ax = plt.subplots()
		else: 
			fig = plt.gcf()
		if vmin is None:
			vmin = self.reduced[self.prop_dim].min()
		if vmax is None:
			vmax = self.reduced[self.prop_dim].max()
		
		ax.scatter(self.reduced['v1'],self.reduced['v2'],c=self.reduced[self.prop_dim],cmap=cmap,vmin=vmin,vmax=vmax,**kwargs)
		if cbar==True:
			add_colorbar(fig=fig,ax=ax,cmap=cmap,label=self.prop_dim, vmin=vmin,vmax=vmax,
				subplots_adjust=dict(left=0.1,right=0.8),cbrect=cbrect)
		ax.set_xlabel('$v_1$')
		ax.set_ylabel('$v_2$')
	
	def reduced_highlight_plot(self,ax=None,cmap=plt.cm.viridis,vmin=None,vmax=None,s=8,cbar=True, cbrect=[0.88,0.12,0.02,0.75],**kwargs):
		"""
		scatter plot of prop_dim in reduced-dimension composition space with outliers highlighted in red
		
		Args:
		-----
		ax: axis on which to plot. if None, create new axis
		cmap: colormap
		vmin: vmin for colormap
		vmax: vmax for colormap
		s: marker size
		cbar: if True, create a colorbar
		cbrect: colorbar rectangle: [left, bottom, width, height]
		kwargs: kwargs to pass to plt.scatter
		"""
		outliers = self.reduced.loc[self.reduced['outlier_flag']==-1,:]
		inliers = self.reduced.loc[self.reduced['outlier_flag']!=-1,:]
		
		if ax is None:
			fig, ax = plt.subplots()
		else: 
			fig = plt.gcf()
		if vmin is None:
			vmin = self.reduced[self.prop_dim].min()
		if vmax is None:
			vmax = self.reduced[self.prop_dim].max()
			
		ax.scatter(inliers['v1'],inliers['v2'],c=inliers[self.prop_dim],cmap=cmap,vmin=vmin,vmax=vmax,s=s,**kwargs)
		ax.scatter(outliers['v1'],outliers['v2'],c=outliers[self.prop_dim],cmap=cmap,vmin=vmin,vmax=vmax,s=s*2,
				   edgecolors='r', linewidths=0.7, **kwargs)
		if cbar==True:
			add_colorbar(fig=fig,ax=ax,cmap=cmap,label=self.prop_dim, vmin=vmin,vmax=vmax,
				subplots_adjust=dict(left=0.1,right=0.8),cbrect=cbrect)
		ax.set_xlabel('$v_1$')
		ax.set_ylabel('$v_2$')
		
		return ax
		
	def reduced_inlier_plot(self,ax=None,cmap=plt.cm.viridis,vmin=None,vmax=None,cbar=True,cbrect=[0.88,0.12,0.02,0.75],**kwargs):
		inliers = self.reduced.loc[self.reduced['outlier_flag']!=-1,:]
		
		if ax is None:
			fig, ax = plt.subplots()
		else: 
			fig = plt.gcf()
		if vmin is None:
			vmin = self.reduced[self.prop_dim].min()
		if vmax is None:
			vmax = self.reduced[self.prop_dim].max()
			
		ax.scatter(inliers['v1'],inliers['v2'],c=inliers[self.prop_dim],cmap=cmap,vmin=vmin,vmax=vmax,**kwargs)
		if cbar==True:
			add_colorbar(fig=fig,ax=ax,cmap=cmap,label=self.prop_dim, vmin=vmin,vmax=vmax,
				subplots_adjust=dict(left=0.1,right=0.8),cbrect=cbrect)
		ax.set_xlabel('$v_1$')
		ax.set_ylabel('$v_2$')
		
		return ax
		
	def reduced_outlier_plot(self,ax=None,cmap=plt.cm.viridis,vmin=None,vmax=None,cbar=True,cbrect=[0.88,0.12,0.02,0.75],**kwargs):
		"""
		scatter plot of prop_dim in reduced-dimension composition space with outliers highlighted in red
		
		Args:
		-----
		kwargs: kwargs to pass to plt.scatter
		"""
		outliers = self.reduced.loc[self.reduced['outlier_flag']==-1,:]
		
		if ax is None:
			fig, ax = plt.subplots()
		else: 
			fig = plt.gcf()
		if vmin is None:
			vmin = self.reduced[self.prop_dim].min()
		if vmax is None:
			vmax = self.reduced[self.prop_dim].max()
			
		ax.scatter(outliers['v1'],outliers['v2'],c=outliers[self.prop_dim],cmap=cmap,vmin=vmin,vmax=vmax,**kwargs)
		if cbar==True:
			add_colorbar(fig=fig,ax=ax,cmap=cmap,label=self.prop_dim, vmin=vmin,vmax=vmax,
				subplots_adjust=dict(left=0.1,right=0.8),cbrect=cbrect)
		ax.set_xlabel('$v_1$')
		ax.set_ylabel('$v_2$')
		
		return ax
	
	def reduced_cluster_plot(self, ax=None,cmap=plt.cm.plasma, cbar=True,cbrect = [0.88,0.12,0.02,0.75], **kwargs):
				
		vmin = self.pred['cluster'].min()
		vmax = self.pred['cluster'].max()
		
		#make norm for discrete colormap
		clusters = list(self.cluster_name.keys())
		cluster_names = list(self.cluster_name.values())
		n_clusters = len(self.pred['cluster'].unique())
		bounds = np.arange(min(clusters)-0.5,max(clusters)+0.51)
		norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
		
		if ax is None:
			fig, ax = plt.subplots()
		else: 
			fig = plt.gcf()
		ax.scatter(self.reduced['v1'],self.reduced['v2'],c=self.reduced['cluster'],cmap=cmap,norm=norm, **kwargs)
		ax.set_xlabel('$v_1$')
		ax.set_ylabel('$v_2$')
		if cbar==True:
			add_colorbar(fig=fig,ax=ax,cmap=cmap,norm=norm,label='Cluster',ticks=clusters,ticklabels=cluster_names,
					 subplots_adjust=dict(left=0.1,right=0.8),cbrect=cbrect)
		
		return ax
		
	def cluster_sample(self):
		if self.cluster_by=='sample':
			return self.cluster_name
		if 'sample' in self.data.columns:
			cluster_sample = {}
			for cluster, cdf in self.data_pred.groupby('cluster'):
				cluster_sample[cluster] = list(cdf['sample'].unique())
			return cluster_sample
		else:
			raise Exception('Data does not contain sample column')
				
		