# Module for matminer-style featurizers

import numpy as np
import pandas as pd
import pymatgen as mg
import collections
from pymatgen.core.composition import Composition
from pymatgen.core.molecular_orbitals import MolecularOrbitals
from calc_chemfeat import Perovskite
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty, ValenceOrbital, CohesiveEnergy
from matminer.featurizers.base import BaseFeaturizer

from warnings import warn

print('loaded featurizers')


class AtomicOrbitalsMod(BaseFeaturizer):
	"""
	*Modified from matminer class to handle cases where LUMO is None*
	Determine HOMO/LUMO features based on a composition.
	The highest occupied molecular orbital (HOMO) and lowest unoccupied
	molecular orbital (LUMO) are estiated from the atomic orbital energies
	of the composition. The atomic orbital energies are from NIST:
	https://www.nist.gov/pml/data/atomic-reference-data-electronic-structure-calculations
	Warning:
	For compositions with inter-species fractions greater than 10,000 (e.g.
	dilute alloys such as FeC0.00001) the composition will be truncated (to Fe
	in this example). In such extreme cases, the truncation likely reflects the
	true physics of the situation (i.e. that the dilute element does not
	significantly contribute orbital character to the band structure), but the
	user should be aware of this behavior.
	"""

	def featurize(self, comp):
		"""
		Args:
			comp: (Composition)
				pymatgen Composition object
		Returns:
			HOMO_character: (str) orbital symbol ('s', 'p', 'd', or 'f')
			HOMO_element: (str) symbol of element for HOMO
			HOMO_energy: (float in eV) absolute energy of HOMO
			LUMO_character: (str) orbital symbol ('s', 'p', 'd', or 'f')
			LUMO_element: (str) symbol of element for LUMO
			LUMO_energy: (float in eV) absolute energy of LUMO
			gap_AO: (float in eV)
				the estimated bandgap from HOMO and LUMO energeis
		"""

		integer_comp, factor = comp.get_integer_formula_and_factor()

		# warning message if composition is dilute and truncated
		if not (len(Composition(comp).elements) ==
				len(Composition(integer_comp).elements)):
			warn('AtomicOrbitals: {} truncated to {}'.format(comp,
															 integer_comp))

		homo_lumo = MolecularOrbitals(integer_comp).band_edges

		feat = collections.OrderedDict()
		
		for edge in ['HOMO', 'LUMO']:
			if homo_lumo[edge] is not None:
				feat['{}_character'.format(edge)] = homo_lumo[edge][1][-1]
				feat['{}_element'.format(edge)] = homo_lumo[edge][0]
				feat['{}_energy'.format(edge)] = homo_lumo[edge][2]
			else:
				#if LUMO is None
				feat['{}_character'.format(edge)] = 'na'
				feat['{}_element'.format(edge)] = 'na'
				#unclear what this value should be. Arbitrarily set to 0. Don't want NaN for modeling
				feat['{}_energy'.format(edge)] = 0 
				
		feat['gap_AO'] = feat['LUMO_energy'] - feat['HOMO_energy']

		return list(feat.values())

	def feature_labels(self):
		feat = []
		for edge in ['HOMO', 'LUMO']:
			feat.extend(['{}_character'.format(edge),
						 '{}_element'.format(edge),
						 '{}_energy'.format(edge)])
		feat.append("gap_AO")
		return feat

	def citations(self):
		return [
			"@article{PhysRevA.55.191,"
			"title = {Local-density-functional calculations of the energy of atoms},"
			"author = {Kotochigova, Svetlana and Levine, Zachary H. and Shirley, "
			"Eric L. and Stiles, M. D. and Clark, Charles W.},"
			"journal = {Phys. Rev. A}, volume = {55}, issue = {1}, pages = {191--199},"
			"year = {1997}, month = {Jan}, publisher = {American Physical Society},"
			"doi = {10.1103/PhysRevA.55.191}, "
			"url = {https://link.aps.org/doi/10.1103/PhysRevA.55.191}}"]
			
	def implementors(self):
		return ['Maxwell Dylla', 'Anubhav Jain']

class PerovskiteProperty(BaseFeaturizer):
	"""
	Class to calculate perovskite features. Includes custom features from the Perovskite class and generic features from ElementProperty, 
	AtomicOrbitals, ValenceOrbital, and CohesiveEnergy matminer featurizers.
	
	Options for initializing:
		ordered_formula_featurizer(): for featurizing ordered formulas
		cation_site_featurizer(): for featurizing unordered formulas based on user-provided cation site assignments
		from_preset(): load a preset
		The class can also be called manually, but be aware that different parameter sets are required for an ordered formula featurizer instance than for a cation site featurizer instance.	
	
	Parameters:
	-----------
	cation_site: dict of site assignments for cations, i.e. {el:site}. Elements not in cation_site are assumed to be anions on X-site
	site_ox_lim: dict of oxidation state limits for each site, i.e. {site:[min,max]}. Elements on sites are limited to oxidation states within these limits
	site_base_ox: dict of base oxidation state for each site, i.e. {site:ox}. Used for determining aliovalent ions and acceptor/donor dopants
	ordered_formulas: if True, determine cation site assignments from order 
	A_site_occupancy: Number of atoms on A site. Used when ordered_formulas is True
	anions: list of anions. Used when ordered_formulas is True
	
	Parameters for ordered formula featurizer: site_ox_lim, site_base_ox, A_site_occupancy, anions
	Parameters for cation site featurizer: cation_site, site_ox_lim, site_base_ox
	"""
	
	def __init__(self, cation_site=None, site_ox_lim={'A':[0,10],'B':[0,10],'X':[-10,0]}, site_base_ox={'A':2,'B':4,'X':-2},
					ordered_formulas=False,A_site_occupancy=1,anions=None):
					
		if cation_site is None and ordered_formulas is False:
			raise ValueError('Either cation sites must be assigned, or formulas must be ordered. Otherwise site assignments can not be determined')
			
		self.cation_site = cation_site
		self.site_ox_lim = site_ox_lim
		self.site_base_ox = site_base_ox
		self.ordered_formulas = ordered_formulas
		self.A_site_occupancy = A_site_occupancy
		self.anions = anions
		
		#matminer featurizers
		self.ValenceOrbital = ValenceOrbital()
		self.AtomicOrbitals = AtomicOrbitalsMod()
		self.CohesiveEnergy = CohesiveEnergy()
		#custom ElementProperty featurizer
		elemental_properties = ['BoilingT', 'MeltingT',
			'BulkModulus', 'ShearModulus', 
			'Row', 'Column', 'Number', 'MendeleevNumber', 'SpaceGroupNumber',
			'Density','MolarVolume',
			'FusionEnthalpy','HeatVaporization',
			'NsUnfilled','NpUnfilled','NdUnfilled','NfUnfilled',
			'Polarizability', 
			'ThermalConductivity']
		self.ElementProperty = ElementProperty(data_source='magpie',features=elemental_properties,
						  stats=["mean", "std_dev", "range"])
		
		self.check_matminer_featurizers()
		self.featurize_options = {}
	
	@classmethod
	def from_preset(cls, preset_name):
		"""
		Initialize from preset
		
		Parameters:
		-----------
		preset_name: name of preset to load. Currently accepts 'BCFZY'
		"""
		if preset_name=='BCFZY':
			#Ba(Co,Fe,Zr,Y)O_3-d system
			cation_site={'Ba':'A','Co':'B','Fe':'B','Zr':'B','Y':'B'}
			site_ox_lim={'A':[2,2],'B':[2,4],'X':[-2,-2]}
			site_base_ox={'A':2,'B':4,'X':-2}
		else:
			raise ValueError("Invalid preset_name specified!")
		
		return cls(cation_site,site_ox_lim, site_base_ox)
		
	@classmethod
	def ordered_formula_featurizer(cls,A_site_occupancy=1,anions=None,site_ox_lim={'A':[0,10],'B':[0,10],'X':[-10,0]}, site_base_ox={'A':2,'B':4,'X':-2}):
		"""
		Convenience method for instantiating a featurizer for ordered formulas
		"""
		return cls(cation_site=None,site_ox_lim=site_ox_lim,site_base_ox=site_base_ox,ordered_formulas=True,A_site_occupancy=A_site_occupancy,anions=anions)
		
	@classmethod
	def cation_site_featurizer(cls,cation_site, site_ox_lim={'A':[0,10],'B':[0,10],'X':[-10,0]}, site_base_ox={'A':2,'B':4,'X':-2}):
		"""
		Convenience method for instantiating a featurizer for unordered formulas, based on site assignments
		"""
		return cls(cation_site,site_ox_lim,site_base_ox)
		
		
	@property
	def ElementProperty_custom_labels(self):
		"""
		Generate custom labels for ElementProperty featurizer that follow same naming convention as Perovskite class
		"""
		elemental_property_label_map = {'BoilingT':'boil_temp','MeltingT':'melt_temp',
							'BulkModulus':'bulk_mod','ShearModulus':'shear_mod',
							'Row':'row','Column':'column','Number':'number','MendeleevNumber':'mendeleev','SpaceGroupNumber':'space_group',
							'Density':'density','MolarVolume':'molar_vol',
							'FusionEnthalpy':'H_fus','HeatVaporization':'H_vap',
							'NsUnfilled':'valence_unfilled_s','NpUnfilled':'valence_unfilled_p','NdUnfilled':'valence_unfilled_d','NfUnfilled':'valence_unfilled_f',
							'Polarizability':'polarizability',
							'ThermalConductivity':'sigma_therm'}
							
		element_property_labels = list(map(elemental_property_label_map.get,self.ElementProperty.features))
		labels = []
		for attr in element_property_labels:
			for stat in self.ElementProperty.stats:
				if stat=='std_dev':
					stat = 'std'
				labels.append(f'{attr}_{stat}')
		return labels
		
	@property
	def ElementProperty_categories(self):
		"""
		Generate categories for ElementProperty featurizer
		"""
		elemental_property_category_map = {'BoilingT':'elemental','MeltingT':'elemental',
							'BulkModulus':'elemental','ShearModulus':'elemental',
							'Row':'periodic','Column':'periodic','Number':'periodic','MendeleevNumber':'periodic','SpaceGroupNumber':'periodic',
							'Density':'elemental','MolarVolume':'elemental',
							'FusionEnthalpy':'elemental','HeatVaporization':'elemental',
							'NsUnfilled':'electronic','NpUnfilled':'electronic','NdUnfilled':'electronic','NfUnfilled':'electronic',
							'Polarizability':'elemental', 
							'ThermalConductivity':'elemental'} 
							
		element_property_categories = list(map(elemental_property_category_map.get,self.ElementProperty.features))
		categories = []
		for ep_cat in element_property_categories:
			for stat in self.ElementProperty.stats:
				categories.append(ep_cat)
		return categories
		
	@property
	def ElementProperty_units(self):
		"""
		Generate units for ElementProperty featurizer
		"""
		elemental_property_unit_map = {'BoilingT':'temp','MeltingT':'temp',
							'BulkModulus':'pressure','ShearModulus':'pressure',
							'Row':'none','Column':'none','Number':'none','MendeleevNumber':'none','SpaceGroupNumber':'none',
							'Density':'density','MolarVolume':'volume',
							'FusionEnthalpy':'energy','HeatVaporization':'energy',
							'NsUnfilled':'none','NpUnfilled':'none','NdUnfilled':'none','NfUnfilled':'none',
							'Polarizability':'polarizability', #complex units - doesn't matter
							'ThermalConductivity':'therm'} #complex units - doesn't matter
							
		element_property_units = list(map(elemental_property_unit_map.get,self.ElementProperty.features))
		units = []
		for ep_unit in element_property_units:
			for stat in self.ElementProperty.stats:
				units.append(ep_unit)
		return units
		
	def ElementProperty_label_check(self):
		"""
		Check that ElementProperty feature labels are as expected
		If not, features may not align with feature labels
		"""
		#ElementProperty.feature_labels() code as of 2/17/19
		labels = []
		for attr in self.ElementProperty.features:
			src = self.ElementProperty.data_source.__class__.__name__
			for stat in self.ElementProperty.stats:
				labels.append("{} {} {}".format(src, stat, attr))
		
		if labels!=self.ElementProperty.feature_labels():
			raise Exception('ElementProperty features or labels have changed')
	
	def set_featurize_options(self,sites,ox_stats=['min','max','mean','median','std','range'],ep_stats=["mean", "std_dev", "range"],radius_type='ionic_radius',normalize_formula=True,silent=True,categories=None):
		"""
		Set options for featurization. Since these options should be the same for all compositions in a batch, set for the featurizer instance rather than passing as args to featurize()
		so that they do not have to be duplicated in every row of a DataFrame when calling featurize_dataframe().
		Since these options change the number and meaning of features returned, it's also safest to set for the whole instance for consistency.
		
		Parameters:
		-----------
		sites: list or string of sites to featurize. Any combination of 'A', 'B', 'X', and/or 'comp' accepted. 
			Composition-level, oxidation-state-dependent features are always calculated by the Perovskite class. Passing '' or [] will return only these features.
			Specifying 'A','B', and/or 'X' sites will calculate site-level features for these sites (oxidation-state independent and dependent features, and matminer features).
			Including 'comp' will calculate oxidation-state-independent features and matminer features for the full composition.
		ox_stats: list of aggregate functions to apply to oxidation state combinations for feature generation using Perovskite class. 
			Options: 'min','max','mean','median','std','range'
		ep_stats: ElementProperty stats. Options: "minimum", "maximum", "range", "mean", "avg_dev", "mode"
		radius_type: Shannon radius type to use in features. Accepts 'crystal_radius' or 'ionic_radius'
		normalize_formula: if True, normalize formula such that higher occupancy cation site has one formula unit (applies to Perovskite class only)
		silent: if False, print informational messages from Perovksite class
		categories: list of feature categories to return. If None, return all. Options: 'bonding','structure','charge','composition','electronic','elemental','periodic'
		"""
		
		feat_options = dict(sites=sites,ox_stats=ox_stats,radius_type=radius_type,normalize_formula=normalize_formula,silent=silent)
		self.featurize_options.update(feat_options)
		self.ElementProperty.stats = ep_stats
		
	def featurize(self,formula):
		"""
		Calculate features
		
		Parameters:
		-----------
		formula: chemical formula string
		
		Returns: list of feature values
		"""
		if self.featurize_options=={}:
			raise Exception('Featurize options have not been set. Use set_featurize_options before featurizing')
		
		if self.ordered_formulas is True:
			pvsk = Perovskite.from_ordered_formula(formula, self.A_site_occupancy, self.anions, site_ox_lim = self.site_ox_lim, site_base_ox = self.site_base_ox, 
												radius_type=self.featurize_options['radius_type'],silent=self.featurize_options['silent'])
		elif self.ordered_formulas is False:
			pvsk = Perovskite(formula, self.cation_site, self.site_ox_lim, self.site_base_ox,self.featurize_options['radius_type'],self.featurize_options['normalize_formula'],
							self.featurize_options['silent'])
			
		pvsk_features = pvsk.featurize(self.featurize_options['sites'],self.featurize_options['ox_stats'])
			
		mm_features = []
		for site in self.featurize_options['sites']:
				
			vo_features = self.ValenceOrbital.featurize(pvsk.site_composition[site]) #avg and frac s, p , d, f electrons
			vo_features += [sum(vo_features[0:3])] #avg total valence electrons
			ao_features = self.AtomicOrbitals.featurize(pvsk.site_composition[site]) #HOMO and LUMO character and energy levels (from atomic orbitals)
			ao_features = [ao_features[i] for i in range(len(ao_features)) if i not in (0,1,3,4)] #exclude HOMO_character,HOMO_element, LUMO_character, LUMO_element - categoricals
			ce_features = self.CohesiveEnergy.featurize(pvsk.site_composition[site],formation_energy_per_atom=1e-10) #avg elemental cohesive energy
			ep_features = self.ElementProperty.featurize(pvsk.site_composition[site]) #elemental property features
			mm_features += vo_features + ao_features + ce_features + ep_features
			
		features = list(pvsk_features) + mm_features
			
		return features
		
	@property
	def matminer_labels(self):
		"""
		Feature labels for matminer-derived features
		"""
		labels = [
			#ValenceOrbital labels
			'valence_elec_s_mean',
			'valence_elec_p_mean',
			'valence_elec_d_mean',
			'valence_elec_f_mean',
			'valence_elec_s_frac',
			'valence_elec_p_frac',
			'valence_elec_d_frac',
			'valence_elec_f_frac',
			'valence_elec_tot_mean',
			#AtomicOrbitals labels
			#'HOMO_character',
			'HOMO_energy',
			#'LUMO_character',
			'LUMO_energy',
			'AO_gap',
			#CohesiveEnergy labels
			'cohesive_energy_mean']
			
		#ElementProperty labels
		labels += self.ElementProperty_custom_labels
		
		return labels
		
	@property
	def matminer_categories(self):
		"""
		Feature categories for matminer-derived features
		"""
		categories = [
			#ValenceOrbital categories
			'electronic',
			'electronic',
			'electronic',
			'electronic',
			'electronic',
			'electronic',
			'electronic',
			'electronic',
			'electronic',
			#AtomicOrbitals categories
			#'HOMO_character',
			'electronic',
			#'LUMO_character',
			'electronic',
			'electronic',
			#CohesiveEnergy categories
			'bonding']
			
		#ElementProperty categories
		categories += self.ElementProperty_categories
		
		return categories
		
	@property
	def matminer_units(self):
		"""
		Feature units for matminer-derived features
		"""
		units = [
			#ValenceOrbital units
			'none',
			'none',
			'none',
			'none',
			'none',
			'none',
			'none',
			'none',
			'none',
			#AtomicOrbitals units
			#'HOMO_character',
			'energy',
			#'LUMO_character',
			'energy',
			'energy',
			#CohesiveEnergy units
			'energy']
			
		#ElementProperty units
		units += self.ElementProperty_units
		
		return units
	
	def feature_labels(self):
		"""
		Get list of feature labels
		"""
		try:
			pvsk_labels = Perovskite.from_preset('BaCoO3','BCFZY',silent=True).feature_labels(self.featurize_options['sites'],self.featurize_options['ox_stats'])
		except KeyError:
			raise Exception('Featurize options have not been set. Use set_featurize_options before accessing feature labels')
		
		mm_labels = []
		for site in self.featurize_options['sites']:
			if site=='comp':
				site_label = 'comp'
			else:
				site_label = f'{site}site'	
			mm_labels += [f'{site_label}_{label}' for label in self.matminer_labels]
			
		return pvsk_labels + mm_labels
		
	def feature_categories(self):
		"""
		Get list of feature categories. For quick filtering
		"""
		try:
			pvsk_categories = Perovskite.from_preset('BaCoO3','BCFZY',silent=True).feature_categories(self.featurize_options['sites'],self.featurize_options['ox_stats'])
		except KeyError:
			raise Exception('Featurize options have not been set. Use set_featurize_options before accessing feature labels')
		
		mm_categories = []
		for site in self.featurize_options['sites']:
			mm_categories += self.matminer_categories
			
		return pvsk_categories + mm_categories
	
	def feature_units(self):
		"""
		Get list of feature labels. For dimensional analysis
		"""
		try:
			pvsk_units = Perovskite.from_preset('BaCoO3','BCFZY',silent=True).feature_units(self.featurize_options['sites'],self.featurize_options['ox_stats'])
		except KeyError:
			raise Exception('Featurize options have not been set. Use set_featurize_options before accessing feature labels')
		
		mm_units = []
		for site in self.featurize_options['sites']:
			mm_units += self.matminer_units
			
		return pvsk_units + mm_units
		
	def check_matminer_featurizers(self):
		"""
		Check that features and feature order for matminer featurizers are as expected
		If features or feature order have changed, featurize() may return unexpected features that do not align with feature_labels()
		"""
		#verify that matminer feature labels haven't changed
		if self.ValenceOrbital.feature_labels() != ['avg s valence electrons',
											 'avg p valence electrons',
											 'avg d valence electrons',
											 'avg f valence electrons',
											 'frac s valence electrons',
											 'frac p valence electrons',
											 'frac d valence electrons',
											 'frac f valence electrons']:
			raise Exception('ValenceOrbital features or labels have changed')
			
		if self.AtomicOrbitals.feature_labels() != ['HOMO_character',
											 'HOMO_element',
											 'HOMO_energy',
											 'LUMO_character',
											 'LUMO_element',
											 'LUMO_energy',
											 'gap_AO']:
			raise Exception('AtomicOrbitals features or labels have changed')

		if self.CohesiveEnergy.feature_labels() != ['cohesive energy']:
			raise Exception('CohesiveEnergy features or labels have changed')
											 
		self.ElementProperty_label_check()
		
	
	