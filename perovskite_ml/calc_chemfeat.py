import numpy as np
import pandas as pd
import pymatgen as mg
from pymatgen.ext.matproj import MPRester
import os
import warnings
import itertools
from math import gcd
import mendeleev as mdl

#load elemental electrical conductivity data
package_dir = os.path.split(os.path.realpath(__file__))[0]
elec_conductivity_df = pd.read_csv(os.path.join(package_dir,'ElementalElectricalConductivity.txt'),sep='\t',skipfooter=1,engine='python')
elec_conductivity = dict(zip(elec_conductivity_df['Symbol'],elec_conductivity_df['Electrical Conductivity (S/cm)']))


class MatProjCalc:
	def __init__(self,oxide_dict={}):
		#dict to specify which oxide to use for metal
		self.oxide_dict = oxide_dict
		#dict to store MX bond energies after calculation. Avoid repeated lookups in MP
		self.calc_MX_bond_energy = {} 
		#dict to store formation enthalpies after looking up
		self.fH_dict = {
				('Ce','gas','exp'):(417.1,'Formation enthalpy for Ce in gas phase includes exp data from phases: gas') #correction to MP entry: fH for Ce gas is negative in MP
					}
		self.mp = MPRester(os.environ['MATPROJ_API_KEY'])
		print("Created MatProjCalc instance")
		
	@property
	def common_anions(self):
		"""List of common anions"""
		return ['N','P','O','S','F','Cl','Br','I']
		
	@property
	def dissocation_energy(self):
		"""
		Bond dissociation energies for gases at 298K in kJ/mol
		Source: https://labs.chem.ucsb.edu/zakarian/armen/11---bonddissociationenergy.pdf
		"""
		return dict(N=945.33,P=490,O=498.34,S=429,F=156.9,Cl=242.58,Br=193.87,I=152.549,H=436.002)
		
	@property
	def mn_combos(self):
		"""
		Possible m-n pairs (m,n up to 4)
		"""
		return [(1,1),(1,2),(1,3),(1,4),(2,1),(2,3),(3,1),(3,2),(3,4),(4,1)]
		
	def possible_ionic_formulas(self,metal,anion,metal_ox_lim=None,anion_ox_state=None):
		"""
		Get possible binary ionic compound formulas for metal-anion pair
		
		Parameters:
		-----------
		metal: metal element symbol
		anion: anion element symbol
		metal_ox_lim: tuple of metal oxidation state limits (min, max)
		anion_ox_state: anion oxidation state. If None, will attempt to find the common oxidation state for the anion 
		"""
		#get common oxidation state for anion
		if anion_ox_state is None:
			anion_ox_state = [ox for ox in mg.Element(anion).common_oxidation_states if ox < 0]
			if len(anion_ox_state) > 1:
				raise Exception(f"Multiple common oxidation states for {anion}. Please specify anion_ox_state")
			else:
				anion_ox_state = anion_ox_state[0]
				
		if metal_ox_lim is None:
			metal_ox_lim = [0,np.inf]
		
		return [f'{metal}{m}{anion}{n}' for m,n in self.mn_combos if m/n <= -anion_ox_state and metal_ox_lim[0] <= -anion_ox_state*n/m <= metal_ox_lim[1]]
		
	def get_fH(self,formula, phase='solid', data_type='exp',silent=True):
		"""
		Get average experimental formation enthalpy for formula and phase
		
		Parameters:
		-----------
		formula: chemical formula string
		phase: phase string. Can be 'solid', 'liquid', 'gas', or a specific solid phase (e.g. 'monoclinic'). If 'solid', returns average across all solid phases
		"""
		#first check for corrected/saved data in fH_dict
		try:
			fH,msg = self.fH_dict[(formula,phase,data_type)]
			if silent==False:
				#print('already calculated')
				print(msg)
		#if no entry exists, look up in MP
		except KeyError:
			results = self.mp.get_data(formula,data_type=data_type)
			if data_type=='exp':
				#results = self.mp.get_exp_thermo_data(formula)
				if phase=='solid':
					phase_results = [r for r in results if r.type=='fH' and r.phaseinfo not in ('liquid','gas')]
				else:
					phase_results = [r for r in results if r.type=='fH' and r.phaseinfo==phase]
				phases = np.unique([r.phaseinfo for r in phase_results])
				fH = [r.value for r in phase_results]
				
			elif data_type=='vasp':
				if phase in ('liquid','gas'):
					raise ValueError('VASP data only valid for solid phases')
				elif phase=='solid':
					#get entry with lowest energy above hull
					srt_results = sorted(results,key=lambda x: x['e_above_hull'])
					phase_results = srt_results[0:1]
				else:
					phase_results = [r for r in results if r['spacegroup']['crystal_system']==phase]
				phases = np.unique([r['spacegroup']['crystal_system'] for r in phase_results])
				n_atoms = mg.Composition(formula).num_atoms
				#DFT formation energies given in eV per atom - need to convert to kJ/mol
				fH = [r['formation_energy_per_atom']*n_atoms*96.485 for r in phase_results]
				
			if len(fH)==0:
				raise LookupError('No {} data for {} in {} phase'.format(data_type,formula,phase))
			maxdiff = np.max(fH) - np.min(fH)
			if maxdiff > 15:
				warnings.warn('Max discrepancy of {} in formation enthalpies for {} exceeds limit'.format(maxdiff,formula))
			fH = np.mean(fH)
			
			msg = 'Formation enthalpy for {} in {} phase includes {} data from phases: {}'.format(formula,phase,data_type,', '.join(phases))
			if silent==False:
				print(msg)
			
			#store value and info message for future lookup
			self.fH_dict[(formula,phase,data_type)] = (fH,msg)
			
		return fH

	def ionic_formula_from_ox_state(self,metal,anion,metal_ox_state,anion_ox_state=None,return_mn=False):
		"""
		Get binary ionic compound formula with reduced integer units based on oxidation state
		
		Parameters:
		-----------
		metal: metal element symbol
		anion: anion element symbol
		metal_ox_state: metal oxidation state
		anion_ox_state: anion oxidation state. If None, will attempt to find the common oxidation state for the anion
		return_mn: if True, return formula units for metal (m) and anion (n)
		
		Returns: chemical formula string MmXn, and m, n if return_mn=True
		"""
		#get common oxidation state for anion
		if anion_ox_state is None:
			anion_ox_state = [ox for ox in mg.Element(anion).common_oxidation_states if ox < 0]
			if len(anion_ox_state) > 1:
				raise Exception(f"Multiple common oxidation states for {anion}. Please specify anion_ox_state")
			else:
				anion_ox_state = anion_ox_state[0]
				
		#formula MmXn
		deno = gcd(metal_ox_state,-anion_ox_state)
		m = -anion_ox_state/deno
		n = metal_ox_state/deno
		formula = '{}{}{}{}'.format(metal,m,anion,n)
		if return_mn==False:
			return formula
		else:
			return formula, m, n
			
	def ox_states_from_binary_formula(self,formula,anion=None,anion_ox_state=None):
		"""
		Determine oxidation states from binary formula.
		Could also use mg.Composition.oxi_state_guesses(), but the logic used is more complex.

		Args:
			formula: chemical formula
			anion: Element symbol of anion. If None, search for common anion
			anion_ox_state: oxidation state of anion. If None, assume common oxidation state
		"""
		comp = mg.Composition(formula)
		if len(comp.elements) != 2:
			raise ValueError('Formula must be binary')
		# determine anion
		if anion is None:
			anion = np.intersect1d([e.name for e in comp.elements],self.common_anions)
			if len(anion) > 1:
				raise ValueError('Found multiple possible anions in formula. Please specify anion')
			elif len(anion)==0:
				raise ValueError('No common anions found in formula. Please specify anion')
			else:
				anion = anion[0]
		metal = np.setdiff1d(comp.elements,mg.Element(anion))[0].name
			
		#get common oxidation state for anion
		if anion_ox_state is None:
			anion_ox_state = [ox for ox in mg.Element(anion).common_oxidation_states if ox < 0]
			if len(anion_ox_state) > 1:
				raise Exception(f"Multiple common oxidation states for {anion}. Please specify anion_ox_state")
			else:
				anion_ox_state = anion_ox_state[0]
				
		metal_ox_state = -comp.get(anion)*anion_ox_state/comp.get(metal)
		
		return {metal:metal_ox_state,anion:anion_ox_state}
		
			
	#get_metal_oxide is obsolete - only keeping for validation (used in perovskite_fH)
	def get_metal_oxide(self,metal,return_mn=False):
		"""
		Choose metal oxide formula
		For now, take the lowest common oxidation state with a corresponding stable oxide
		"""
		try: #oxide_dict specifies which oxide to use
			oxide = self.oxide_dict[metal]
			oxide_mg = mg.Composition(oxide)
			m = oxide_mg.get(metal)
			n = oxide_mg.get('O')
			obe = self.MX_bond_energy(oxide)
		except KeyError: #if no oxide indicated in oxide_dict
			"placeholder - for now, take the lowest common oxidation state with a corresponding stable oxide"
			i = 0
			while i != -1: 
				met_mg = mg.Element(metal)
				ox = met_mg.common_oxidation_states[i]
				oxide, m ,n = self.ionic_formula_from_ox_state(metal,'O',ox,return_mn=True)
				try:
					obe = self.MX_bond_energy(oxide)
					#print(obe)
					i = -1
				except LookupError as err:
					i += 1 #try the next oxidation state
					
		#store chosen oxide in oxide_dict
		self.oxide_dict[metal] = oxide
		
		if return_mn==False:
			return oxide
		else:
			return oxide, m, n

	def MX_bond_energy(self,formula,data_type='exp',ordered_formula=False,silent=True):
		"""
		Get metal-anion bond energy per mole of metal for binary ionic compound
		
		Parameters:
		-----------
		formula: chemical formula string
		ordered_formula: if true, assume that first element in formula is metal, and second is anion (i.e. MmXn)
		"""
		
		comp = mg.Composition(formula)
		formula = comp.reduced_formula
		try:
			#look up compound if already calculated
			abe,msg = self.calc_MX_bond_energy[(formula,data_type)]
			if silent==False:
				#print('already calculated')
				print(msg)
		except KeyError:
			if len(comp.elements) != 2:
				raise Exception("Formula is not a binary compound")
				
			if ordered_formula is False:
				anions = [el.name for el in comp.elements if el.name in self.common_anions]
				if len(anions) == 0:
					raise Exception('No common anions found in formula. Use ordered formula to indicate metal and anion')
				elif len(anions) > 1:
					raise Exception('Multiple anions found in formula.  Use ordered formula to indicate metal and anion')
				else:
					anion = anions[0]
				metal = [el.name for el in comp.elements if el.name!=anion][0]
			elif ordered_formula is True:
				metal = comp.elements[0].name
				anion = comp.elements[1].name
				
			m = comp.get_el_amt_dict()[metal]
			n = comp.get_el_amt_dict()[anion]
				
			fH = self.get_fH(formula,data_type=data_type,silent=silent) #oxide formation enthalpy
			H_sub = self.get_fH(metal, phase='gas',silent=silent) #metal sublimation enthalpy - must be exp data (no vasp data for gas)
			#look up info messages from get_fH to store in dict
			msg = self.fH_dict[formula,'solid',data_type][1] + '\n'
			msg += self.fH_dict[metal,'gas','exp'][1]
			DX2 = self.dissocation_energy[anion] #anion dissociation energy
			abe = (fH - m*H_sub - (n/2)*DX2)/m #M-O bond energy per mole of M
			self.calc_MX_bond_energy[(formula,data_type)] = (abe,msg)
		return abe
		
	#perovskite_fH is obsolete - only keeping for validation
	def perovskite_fH(self,formula,A_site,B_site,silent=True):
		"""
		Estimate formation enthalpy of perovskite oxide from simple oxide thermo data
		Uses concept of average M-O bond energy from Sammells et al. (1992), Solid State Ionics 52, 111-123.
		
		Parameters:
		-----------
		formula: oxide formula
		A_site: list of A-site elements
		B_site: list of B-site elements
		verbose: if True, print info about which simple oxides used in calculation
		"""
		comp = mg.Composition(formula)
		cd = comp.get_el_amt_dict()
		metals = A_site + B_site
		
		if silent==False:
			print(f'Oxides used for {formula} fH calculation:')
			
		obe = 0 #total M-O bond enthalpy
		H_sub = 0 #total metal sublimation enthalpy
		
		for metal in metals:
			amt = cd[metal]
			met_mg = mg.Element(metal)
			
			oxide, m, n = self.get_metal_oxide(metal,return_mn=True)
			
			if silent==False:
				print(oxide)
				
			obe += self.MX_bond_energy(oxide)*amt
			H_sub += self.get_fH(metal,'gas')*amt
		
		fH = H_sub + obe + 1.5*self.dissocation_energy['O']
		return fH

#create MatProjCalc instance to store fetched data/calculations
mpcalc = MatProjCalc()
	
class Perovskite:
	"""
	Class to calculate perovskite features
	
	Can be initialized using from_preset() method
	
	Parameters:
	-----------
	formula: chemical formula string. X-site units should be nominal (i.e., delta=0)
	cation_site: dict of site assignments for cations, i.e. {el:site}. Elements not in cation_site are assumed to be anions on X-site
	site_ox_lim: dict of oxidation state limits for each site, i.e. {site:[min,max]}. Elements on sites are limited to oxidation states within these limits
	site_base_ox: dict of base oxidation state for each site, i.e. {site:ox}. Used for determining aliovalent ions and acceptor/donor dopants
	radius_type: Shannon radius type to use in features. Accepts 'crystal_radius' or 'ionic_radius'
	normalize_formula: if True, normalize formula such that higher occupancy cation site has one formula unit
	silent: if False, print informational messages
	"""
	
	def __init__(self, 
				 formula,
				 cation_site={'Ba':'A','Co':'B','Fe':'B','Zr':'B','Y':'B'},
				 site_ox_lim={'A':[0,10],'B':[0,10],'X':[-10,0]},
				 site_base_ox={'A':2,'B':4,'X':-2},
				 radius_type='ionic_radius',
				 normalize_formula=True,
				 silent=True):
		if normalize_formula==True:
			# normalize higher occupancy cation site to 1
			comp = mg.Composition(formula)
			cation_amt_dict = {k:v for k,v in comp.get_el_amt_dict().items() if k in cation_site.keys()}
			A_sum = sum(amt for el,amt in cation_amt_dict.items() if cation_site[el]=='A')
			B_sum = sum(amt for el,amt in cation_amt_dict.items() if cation_site[el]=='B')
			factor = max(A_sum,B_sum)
			norm_comp = mg.Composition({el:amt/factor for el,amt in comp.get_el_amt_dict().items()})
			formula = norm_comp.formula
			
		self.formula = formula
		self.site_ox_lim = site_ox_lim
		self.site_base_ox = site_base_ox
		self.radius_type = radius_type
		self.silent=silent
		self.composition = mg.Composition(formula)
		self.el_amt_dict = self.composition.get_el_amt_dict()
		
		#remove cations not in formula from cation_site dict
		rem = [c for c in cation_site.keys() if c not in self.el_amt_dict.keys()]
		cs = cation_site.copy()
		for r in rem:
			del cs[r]
		self.cation_site = cs
		
		#set site compositions
		cation_amt_dict = {k:v for k,v in self.el_amt_dict.items() if k in cation_site.keys()}
		A_composition = mg.Composition.from_dict({k:v for k,v in cation_amt_dict.items() if cation_site[k]=='A'})
		B_composition = mg.Composition.from_dict({k:v for k,v in cation_amt_dict.items() if cation_site[k]=='B'})
		X_composition = mg.Composition.from_dict({k:v for k,v in self.el_amt_dict.items() if k not in cation_site.keys()})
		self.site_composition = {'A':A_composition,'B':B_composition,'X':X_composition,'comp':self.composition}

		# set nominal site amounts
		AB_max = max([self.site_sum('A'),self.site_sum('B')])
		self.nominal_site_amt = {'A':AB_max,'B':AB_max,'X':AB_max*3}
		
		#create A_site and B_site lists for convenience
		# self.A_site = [c for c in self.cations if self.cation_site[c]=='A']
		# self.B_site = [c for c in self.cations if self.cation_site[c]=='B']
		
		#create ion_site dict for site assignments
		self.ion_site = self.cation_site.copy()
		anion_site = {el.name:'X' for el in X_composition.elements}
		self.ion_site.update(anion_site)
		
		#initialize cation_ox_lim dict with site limits
		self._ion_ox_lim = {}
		for ion, site in self.ion_site.items():
			self.set_ion_ox_lim(ion,self.site_ox_lim.get(site,[-10,10]))
		
		#initialize site_feature dicts
		self._site_features = {}
		self._site_ox_features = {}
		self.site_MX_ABE = {}
		
		#run checks
		for site in 'ABX':
			#ensure all sites have constituent ions
			if len(self.site_composition[site].elements)==0:
				raise Exception(f'{site} site is empty. Check formula and cation_site entries')
			#ensure valid oxidation state limits (no limit is valid)
			ox_lim = self.site_ox_lim.get(site,[0,0])
			if len(ox_lim) != 2 or ox_lim[0] > ox_lim[1]:
				raise Exception(f"Invalid oxidation state limits {ox_lim} for {site} site")
		if self.radius_type not in ('crystal_radius','ionic_radius'):
			raise Exception(f'Invalid radius type {self.radius_type}. Options are crystal_radius and ionic_radius')
	
	@classmethod
	def from_preset(cls,formula,preset_name,radius_type='crystal_radius',normalize_formula=False,silent=True):
		if preset_name=='BCFZY':
			#Ba(Co,Fe,Zr,Y)O_3-d system
			cation_site={'Ba':'A','Co':'B','Fe':'B','Zr':'B','Y':'B'}
			site_ox_lim={'A':[2,2],'B':[2,4],'X':[-2,-2]}
			site_base_ox={'A':2,'B':4,'X':-2}
		else:
			raise ValueError("Invalid preset_name specified!")
		
		return cls(formula,cation_site,site_ox_lim,site_base_ox,radius_type,normalize_formula,silent)
		
		
	@classmethod
	def from_ordered_formula(cls,formula,A_site_occupancy=1,anions=None,**kw):
		"""
		Instantiate from ordered formula. Automatically determine cation site assignments 
		Ordered formula should list A-site cations first, B-site cations second, and anions last
		
		Args:
			formula: chemical_formula
			A_site_occupancy: number of atoms on A site
			anions: list of anions in formula. If None, defaults to common anions
			kw: kwargs to pass to Perovskite (site_ox_llim,site_base_ox, radius_type, silent)
		"""
		
		comp = mg.Composition(formula)
		if anions is None:
			anions = mpcalc.common_anions
		
		A_amt = 0
		cation_site = {}
		for el,amt in comp.get_el_amt_dict().items():
			if el not in anions:
				if round(A_amt,5) < A_site_occupancy:
					cation_site[el] = 'A'
					A_amt += amt
					if round(A_amt,5) > A_site_occupancy:
						raise ValueError(f'Cations cannot be assigned without splitting {el} between sites')
				elif round(A_amt,5) == A_site_occupancy:
					cation_site[el] = 'B'
				elif round(A_amt,5) > A_site_occupancy:
					# this should be caught above, but just in case I missed something
					raise ValueError('Too many atoms on A site')
		
		# if site_ox_lim is None:
			# for site in 'AB':
				# elements = [v for k,v in cation_site.items() if v==site]
				# ox_states = sum([list(mg.Element(el).common_oxidation_states) for el in elements],[])
				# ox_states = [ox for ox in ox_states if ox > 0]
				# site_ox_lim[site] = [min(ox_states),max(ox_states)]
				
		# if site_base_ox is None:
			# this is used only to determine aliovalent_ions, which is used only to calculate alio_net_mag
			
				
		
		return cls(formula,cation_site,normalize_formula=False,**kw)
		
		
	@property
	def site_CN(self):
		'''
		Standard coordination numbers for each site
		'''
		return {'A':12,'B':6,'X':2}
	
	def get_ion_ox_lim(self):
		'''
		getter for ion_ox_lim
		'''
		return self._ion_ox_lim
	
	def set_ion_ox_lim(self,cat,lim):
		'''
		setter for ion_ox_lim
		'''
		self._ion_ox_lim[cat] = lim
		#print('Set lim')
		
	ion_ox_lim = property(get_ion_ox_lim,set_ion_ox_lim, 
						  doc='''
						  Oxidation state limits for each ion 
						  (doesn\'t reflect physically allowed oxidation states,
						  just user-defined limits on the range of physical oxidation states that will be considered)''')
	
	@property
	def cations(self):
		'''
		cations in formula
		'''
		return list(self.cation_site.keys())
	
	def site_sum(self,site):
		'''
		total site formula units
		
		Parameters:
		-----------
		site: 'A','B', or 'X'
		'''
		return self.site_composition[site].num_atoms
	
	def site_mean_mg_prop(self,site,property_name): 
		'''
		general function for averaging properties in mg.Element data dict for elements on a site
		'''
		return np.average([el.data[property_name] for el in self.site_composition[site].elements],weights=list(self.site_composition[site].get_el_amt_dict().values()))
	
	def site_mean_func(self,site,func,**kwargs):
		'''
		general function for averaging the value of a function "func" for elements on a site
		'''
		return np.average([func(el) for el in self.site_composition[site].elements],weights=list(self.site_composition[site].get_el_amt_dict().values()))
			
	@property
	def allowed_ox_states(self):
		"""
		returns allowed cation oxidation states as dict of tuples
		"""
		ox_states = dict()
		for el in self.composition.elements:
			if len(el.common_oxidation_states)==1: #if only 1 common ox state, choose that
				ox_states[el.name] = el.common_oxidation_states
			else: #otherwise take ox states corresponding to Shannon radii
				oxlim = self.ion_ox_lim[el.name]
				ox_states[el.name] = tuple([int(x) for x in el.data['Shannon radii'].keys() 
									  if oxlim[0] <= int(x) <= oxlim[1]])
		return ox_states
	
	@property 
	def multivalent_ions(self):
		'''
		cations with multiple allowed oxidation states
		'''
		return [el for el in self.composition.elements if len(self.allowed_ox_states[el.name]) > 1]
		
	@property
	def aliovalent_ions(self):
		'''
		Dict of aliovalent ions and their valence differences from the base oxidation state for the site they occupy
		'''
		#only consider single-valence ions
		#don't treat multivalent ions as aliovalent
		alio = {}
		for el,site in self.ion_site.items():
			if len(self.allowed_ox_states[el])==1:
				if self.allowed_ox_states[el][0] != self.site_base_ox[site]:
					alio[el] = self.allowed_ox_states[el][0] - self.site_base_ox[site]
		return alio
	
	@property
	def acceptors(self):
		'''
		Dict of acceptor dopants and their valence differences from the base oxidation state for the site they occupy
		'''
		return {k:v for k,v in self.aliovalent_ions.items() if v < 0}
	
	@property
	def donors(self):
		'''
		Dict of donor dopants and their valence differences from the base oxidation state for the site they occupy
		'''
		return {k:v for k,v in self.aliovalent_ions.items() if v > 0}
		
	@property
	def acceptor_mag(self): 
		'''
		Magnitude of acceptor doping (acceptor amount*valence delta)
		'''
		return np.sum([self.el_amt_dict[a]*self.acceptors[a] for a in self.acceptors.keys()])
	
	@property
	def donor_mag(self): 
		'''
		Magnitude of donor doping (donor amount*valence delta)
		'''
		return np.sum([self.el_amt_dict[a]*self.donors[a] for a in self.donors.keys()])
	
	@property
	def alio_net_mag(self): 
		'''
		Net magnitude of aliovalent doping (amount*valence delta)
		'''
		return np.sum([self.el_amt_dict[a]*self.aliovalent_ions[a] for a in self.aliovalent_ions])
	
	@property
	def _roman_to_int(self): 
		'''
		Roman numeral to integer dict 
		Needed for Shannon radii coordination numbers
		'''
		return {'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,'VIII':8,'IX':9,'X':10,'XI':11,'XII':12}
	
	@property
	def _int_to_roman(self):
		'''
		Integer to roman numeral dict
		'''
		return {v:k for k,v in self._roman_to_int.items()}
	
	def _closest_CN(self, el, ox, target_CN): 
		'''
		Get closest coordination number (in roman numerals) to target CN
		Convenience function for choosing appropriate Shannon radius - used in ox_combos
		
		Parameters:
		-----------
		el: pymatgen Element
		ox: oxidation state
		target_CN: target coordination number
		'''
		#get possible CNs for oxidation state
		cn_rom = el.data['Shannon radii'][f'{ox}'].keys()
		#remove any roman numerals not in dict (eg "IVSQ")
		cn_rom = [rn for rn in cn_rom if rn in self._roman_to_int.keys()] 
		#convert roman numerals to ints
		cn = np.array([self._roman_to_int[rn] for rn in cn_rom])
		#find CNs with min difference from target CN
		cn = cn[abs(cn-target_CN) == min(np.abs(cn-target_CN))]
		#convert back to roman numerals
		rom = [self._int_to_roman[c] for c in cn]
		return rom
	
	"""
	Old logic for MX_ABE (commented) - for each possible oxidation state, calculate ABE for corresponding ionic compound.
	Problem with this approach: miss compounds with fractional avg oxidation states, e.g. Co3O4
	New logic below
	"""
	# def site_MX_ABE(self,site,ion_ox_states):
		# """
		# Get average metal-anion bond energy per mole of metal for cation site
		
		# Parameters:
		# -----------
		# site: 'A' or 'B' ('X' invalid since X-X bonds are not relevant)
		# ion_ox_states: dict of ion oxidation states {el:ox}
		# """
		# if site in 'AB':
			# composition = self.site_composition[site]
		# elif site in ('X','comp'):
			# raise Exception('M-X bond energy is only valid for A and B sites')
		# else:
			# raise Exception(f'Site {site} not recognized')
			
		# if len(self.site_composition['X'].elements)==1:
			# #if single anion, get average MX bond energy for site
			# anion = self.site_composition['X'].elements[0].name
			# binaries = [mpcalc.ionic_formula_from_ox_state(el.name,anion,ion_ox_states[el.name]) for el in composition.elements]
			# energies = [mpcalc.MX_bond_energy(binary,silent=self.silent) for binary in binaries]
			# abe = np.average(energies,weights=list(composition.get_el_amt_dict().values()))
		# else:
			# #if multiple anions, get average MX bond energy for each anion, then get weighted average across anions
			# anion_abe = np.empty(len(self.site_composition['X'].elements))
			# for i,anion in enumerate([el.name for el in self.site_composition['X'].elements]):
				# binaries = [mpcalc.ionic_formula_from_ox_state(el.name,anion,ion_ox_states[el.name]) for el in composition.elements]
				# energies = [mpcalc.MX_bond_energy(binary,silent=self.silent) for binary in binaries]
				# anion_abe[i] = np.average(energies,weights=list(composition.get_el_amt_dict().values()))
			# abe = np.average(anion_abe,weights=list(self.site_composition['X'].get_el_amt_dict().values()))
	
		# return abe


	"""
	New MX_ABE logic: find ABE for all binaries within ox limits for each metal, then take cartesian product across metals to get ox combos and find ox_stats for each site
	This gives the site-level MX_ABE features. To get composition-level features, all metals on all sites to get composition MX_ABEs and then get ox_stats
	Also need to take a similar approach for composition formation energy
	"""
	def site_element_MX_ABE(self,site):
		"""
		Get experimental M-X bond energies for binary MmXn compounds for each metal on the specified site.
		Any binary compound found in MatProj that falls within the oxidation state limits will be considered.
		*Note: not currently set up to handle multivalent anions, or anion doping. This should be rare...
			This would break (adding as I notice): 
				anion_delta calculation
				H_formation calculation - requires actual anion amount
				more...?
		
		Parameters:
		-----------
		site: 'A' or 'B' ('X' invalid since X-X bonds are not relevant)
		
		Returns: dict of metals and their calculated ABEs, i.e. {metal:[ABEs]}
		"""
		if site in 'AB':
			composition = self.site_composition[site]
		elif site in ('X','comp'):
			raise Exception('M-X bond energy is only valid for A and B sites')
		else:
			raise Exception(f'Site {site} not recognized')
			
		if len(self.site_composition['X'].elements)==1:
			anion = self.site_composition['X'].elements[0].name
			#get ABEs for all possible binaries for each metal within ox limits
			metal_ABEs = {}
			for metal in [el.name for el in self.site_composition[site].elements]:
				binaries = mpcalc.possible_ionic_formulas(metal,anion,metal_ox_lim=[min(self.allowed_ox_states[metal]),max(self.allowed_ox_states[metal])])
				energies = {}
				for binary in binaries:
					try:
						e = mpcalc.MX_bond_energy(binary,silent=self.silent)
						ox = mpcalc.ox_states_from_binary_formula(binary,anion=anion)[metal]
						energies[ox] = e
					except LookupError:
						#no thermo data for binary compound
						pass
				metal_ABEs[metal] = energies
		
		else:
			raise Exception('Not set up for anion doping yet')
			
		return metal_ABEs
		
	def set_site_oxcombo_MX_ABE(self,site):
		"""
		Calculate and store site-averaged M-X bond energies for all combinations of allowed oxidation states for site
		
		Parameters:
		-----------
		site: 'A' or 'B' ('X' invalid since X-X bonds are not relevant)
		
		Generates dict of site-averaged oxidation state:MX_ABE values. Each value represents one possible combination of oxidation states
		"""
		#get all MX_ABEs within oxidation state limits for each metal
		metal_ABEs = self.site_element_MX_ABE(site)
		ABEs = [list(d.values()) for d in metal_ABEs.values()] # list(metal_ABEs.values())
		OSs = [list(d.keys()) for d in metal_ABEs.values()] # oxidation states
		
		#for each combination of oxidation states, get site-averaged ABE
		composition = self.site_composition[site]
		weights = list(composition.get_el_amt_dict().values())
		mean_ABEs = {}
		for etup,oxtup in zip(itertools.product(*ABEs),itertools.product(*OSs)):
			mean_ABEs[(np.average(oxtup,weights=weights))] = (np.average(etup,weights=weights))
			
		self.site_MX_ABE[site] = mean_ABEs
		
		
	def get_exp_thermo_features(self,sites,ox_stats):
		"""
		Calculate features from experimental thermo data for specified sites. Current features are M-X bond energy (any site) and perovskite formation enthalpy (full composition only).
		Composition-level features are always calculated even if 'comp' is not specified in sites.
		Values for different oxidation state combinations are aggregated into features using the functions specified in ox_stats
		
		Parameters:
		-----------
		sites: list or string of sites to featurize. Any combination of 'A', 'B', 'X', and/or 'comp' accepted. 
			'X' may be passed but will be ignored, as M-X bond energy does not apply to the X site.
		ox_stats: list of aggregate functions to apply to oxidation state combinations for feature generation. 
			Options: 'min','max','mean','median','std','range', or any built-in numpy aggregate function
		
		Returns: dict of features, i.e. {feature_label:feature_value}
		"""
		#get list of possible MX_ABEs for each site
		for site in 'AB':
			self.set_site_oxcombo_MX_ABE(site)
			
		#get composition-averaged MX_ABE and total ABE for each possible combination
		a = self.site_sum('A')
		b = self.site_sum('B')
		compavg_ABEs = {} # {tot_cat_charge:avg_ABE} # np.empty(len(self.site_MX_ABE['A'])*len(self.site_MX_ABE['B']))
		tot_ABEs = {} #{tot_cat_charge:tot_ABE} # np.empty(len(self.site_MX_ABE['A'])*len(self.site_MX_ABE['B']))
		for i, ((ABE_a,ABE_b),(OS_a,OS_b)) in enumerate(zip(itertools.product(*[self.site_MX_ABE['A'].values(),self.site_MX_ABE['B'].values()]), 
												itertools.product(*[self.site_MX_ABE['A'].keys(),self.site_MX_ABE['B'].keys()]))):
			#composition-averaged ABE - Sammells 1992
			compavg_ABEs[a*OS_a + b*OS_b] = 2*np.average([ABE_a/12, ABE_b/6],weights=[a,b])
			#total ABE (for estimating perovskite formation enthalpy)
			tot_ABEs[a*OS_a + b*OS_b] = a*ABE_a + b*ABE_b
		
		self.site_MX_ABE['comp'] = compavg_ABEs #list(compavg_ABEs)
		
		#get total metal sublimation enthalpy
		H_sub = 0
		for el in self.cations:
			H_sub += self.composition.get(el)*mpcalc.get_fH(el,'gas')
			
		# estimate perovskite formation enthalpy for each oxidation state combo
		H_formation = []
		for tot_cat_charge, tot_ABE in tot_ABEs.items():		
			#get total anion dissociation enthalpy
			"""anion_delta is considered in this calculation - assumes that anion vacancy amounts are proportional to nominal anion amounts"""
			tot_anion_amt = -(tot_cat_charge/self.site_base_ox['X'])
			DX2 = 0
			for el,amt in self.site_composition['X'].get_el_amt_dict().items():
				DX2 += tot_anion_amt*(amt/self.site_sum('X'))*mpcalc.dissocation_energy[el]/2
		
			H_f = tot_ABE + H_sub + DX2
			H_formation.append(H_f)
		
		#aggregate list of values for each specified site into features using ox_stats
		#always get the composition-averaged MX_ABE and estimated perovskite formation enthalpy 
		#Get average MX_ABE for A and B sites if specified, but not X site
		if type(sites)==str:
			#if sites given as string (e.g. 'ABX'), split into list for np.intersect1d
			sites = list(sites)
		feature_sites = ['comp'] + list(np.intersect1d(sites,['A','B']))
		#MX_ABE for each site, and perovskite formation enthalpy for full comp only
		features = np.empty((len(feature_sites)+1)*len(ox_stats))
		
		#aggregate formation enthalpies
		for i,stat in enumerate(ox_stats):
			if stat=='range':
				stat_feat = (max(H_formation) - min(H_formation))
			else:
				stat_feat = getattr(np,stat)(H_formation)
			features[i] = stat_feat
		
		#aggregate MX_ABEs
		for i, site in enumerate(feature_sites):
			ABEs = list(self.site_MX_ABE[site].values())
			for j,stat in enumerate(ox_stats):
				if stat=='range':
					stat_feat = (max(ABEs) - min(ABEs))
				else:
					stat_feat = getattr(np,stat)(ABEs)
				features[(i+1)*len(ox_stats) + j] = stat_feat
		
		labels = self.exp_thermo_feature_labels(sites,ox_stats)
		
		return dict(zip(labels,features))
		
	def exp_thermo_feature_labels(self,sites,ox_stats):
		"""
		Get labels for features based on experimental thermo data
		
		Parameters:
		-----------
		sites: list or string of sites to featurize. Any combination of 'A', 'B', 'X', and/or 'comp' accepted. 
			'X' may be passed but will be ignored, as M-X bond energy does not apply to the X site.
		ox_stats: list of aggregate functions to apply to oxidation state combinations for feature generation. 
			Options: 'min','max','mean','median','std','range', or any built-in numpy aggregate function
		
		Returns: list of feature labels
		"""
		if type(sites)==str:
			#if sites given as string (e.g. 'ABX'), split into list for np.intersect1d
			sites = list(sites)
		feature_sites = ['comp'] + list(np.intersect1d(sites,['A','B']))
		
		#MX_ABE for each site, and perovskite formation enthalpy for full comp only
		labels = np.empty((len(feature_sites)+1)*len(ox_stats),dtype='<U50')
		
		#formation enthalpy features
		for i,stat in enumerate(ox_stats):
			labels[i] = f'comp_ox{stat}_H_formation'
		
		#MX_ABE features
		for i, site in enumerate(feature_sites):
			if site=='comp':
				site_label = 'comp'
			else:
				site_label = f'{site}site'
				
			for j,stat in enumerate(ox_stats):
				labels[(i+1)*len(ox_stats) + j] = f'{site_label}_ox{stat}_MX_ABE'
		return labels
		
	def exp_thermo_feature_categories(self,sites,ox_stats):
		"""
		Get categories for features based on experimental thermo data
		
		Parameters:
		-----------
		sites: list or string of sites to featurize. Any combination of 'A', 'B', 'X', and/or 'comp' accepted. 
			'X' may be passed but will be ignored, as M-X bond energy does not apply to the X site.
		ox_stats: list of aggregate functions to apply to oxidation state combinations for feature generation. 
			Options: 'min','max','mean','median','std','range', or any built-in numpy aggregate function
		
		Returns: list of feature units
		"""
		labels = self.exp_thermo_feature_labels(sites,ox_stats)
		return ['bonding']*len(labels)
		
	def exp_thermo_feature_units(self,sites,ox_stats):
		"""
		Get units for features based on experimental thermo data
		
		Parameters:
		-----------
		sites: list or string of sites to featurize. Any combination of 'A', 'B', 'X', and/or 'comp' accepted. 
			'X' may be passed but will be ignored, as M-X bond energy does not apply to the X site.
		ox_stats: list of aggregate functions to apply to oxidation state combinations for feature generation. 
			Options: 'min','max','mean','median','std','range', or any built-in numpy aggregate function
		
		Returns: list of feature units
		"""
		if type(sites)==str:
			#if sites given as string (e.g. 'ABX'), split into list for np.intersect1d
			sites = list(sites)
		feature_sites = ['comp'] + list(np.intersect1d(sites,['A','B']))
		
		#MX_ABE for each site, and perovskite formation enthalpy for full comp only
		units = np.empty((len(feature_sites)+1)*len(ox_stats),dtype='<U50')
		
		#formation enthalpy features
		for i,stat in enumerate(ox_stats):
			units[i] = 'energy'
		
		#MX_ABE features
		for i, site in enumerate(feature_sites):
			if site=='comp':
				site_label = 'comp'
			else:
				site_label = f'{site}site'
				
			for j,stat in enumerate(ox_stats):
				units[(i+1)*len(ox_stats) + j] = 'energy'
		return units
		
	# oxcombo_formation_enthalpies not actually used in featurize()... the logic is integrated into get_exp_thermo_features. Still keeping it for convenience and validation
	# def oxcombo_formation_enthalpies(self):
		# """
		# Get estimated perovskite formation enthalpy for each possible oxidation state combination
		# Estimate uses concept of average M-O bond energy from Sammells et al. (1992), Solid State Ionics 52, 111-123.
		# """
		# #get total metal sublimation enthalpy
		# H_sub = 0
		# for el in self.cations:
			# H_sub += self.composition.get(el)*mpcalc.get_fH(el,'gas')
			
		# #get total anion dissociation enthalpy
		# """thought: should anion_delta be considered in this calculation?"""
		# DX2 = 0
		# for el,amt in self.site_composition['X'].get_el_amt_dict().items():
			# DX2 += amt*mpcalc.dissocation_energy[el]/2
			
		# #get sum of ABEs for each oxidation state combo
		# a = self.site_sum('A')
		# b = self.site_sum('B')
		
		# MX_ABEs = np.empty(len(self.site_MX_ABE['A'])*len(self.site_MX_ABE['B']))
		# for i, (ABE_a, ABE_b) in enumerate(itertools.product(*[self.site_MX_ABE['A'],self.site_MX_ABE['B']])):
			# MX_ABEs[i] = a*ABE_a + b*ABE_b
		
		# H_formation = MX_ABEs + H_sub + DX2
		
		# return H_formation
		
				
	def site_feature_labels(self,site):
		"""
		List of labels for site-level oxidation-state-independent features
		
		Parameters:
		-----------
		site: 'A', 'B', 'X', or 'comp' (for full composition)
		"""
		features = ['num_atoms','X_mean','X_std','TM_frac','multivalent_frac','net_alio_mag','acceptor_mag','donor_mag','mass_mean','mass_std']
		if site in 'AB':
			features += ['MX_IC_mean','sigma_elec_mean']
		return features
		
	def site_feature_categories(self,site):
		"""
		List of categories for site-level oxidation-state-independent features
		
		Parameters:
		-----------
		site: 'A', 'B', 'X', or 'comp' (for full composition)
		"""
		categories = ['composition','bonding','bonding','composition','composition','composition','composition','composition','elemental','elemental']
		if site in 'AB':
			categories += ['bonding','elemental']
		return categories
		
	def site_feature_units(self,site):
		"""
		List of units for site-level oxidation-state-independent features
		
		Parameters:
		-----------
		site: 'A', 'B', 'X', or 'comp' (for full composition)
		"""
		units = ['none','none','none','none','none','none','none','none','mass','mass']
		if site in 'AB':
			units += ['none','S/cm']
		return units
		
	def set_site_features(self,site):
		"""
		Calculate oxidation-state-independent features for site
		
		Parameters:
		-----------
		site: 'A', 'B', 'X', or 'comp' (for full composition)
		"""
		
		try:
			composition = self.site_composition[site]
		except KeyError:
			raise Exception(f'Site {site} not recognized')
		
		features = {}
		features['num_atoms'] = composition.num_atoms
		X_mean = self.site_mean_mg_prop(site,'X')
		features['X_mean'] = X_mean
		features['X_std'] = self.site_mean_func(site,lambda el: (el.X - X_mean)**2)**0.5
		features['TM_frac'] = self.site_mean_func(site,lambda el: el.is_transition_metal)
		features['multivalent_frac'] = self.site_mean_func(site,lambda el: np.sign(len(self.allowed_ox_states[el.name])-1))
		features['net_alio_mag'] = self.site_mean_func(site,lambda el: self.aliovalent_ions.get(el.name,0))
		features['acceptor_mag'] = self.site_mean_func(site,lambda el: self.acceptors.get(el.name,0))
		features['donor_mag'] = self.site_mean_func(site,lambda el: self.donors.get(el.name,0))
		mass_mean = self.site_mean_mg_prop(site,'Atomic mass')
		features['mass_mean'] = mass_mean
		features['mass_std'] = self.site_mean_func(site,lambda el: (el.atomic_mass - mass_mean)**2)**0.5
		#only calculate M-X bond ionic character and electrical conductivity for A and B sites
		if site in 'AB':
			features['MX_IC_mean'] = 1 - np.exp(-0.25*(X_mean - self.site_mean_mg_prop('X','X'))**2)
			features['sigma_elec_mean'] = self.site_mean_func(site,lambda el: elec_conductivity[el.name])/(1e5) #10^5 S/cm
		
		self._site_features[site] = features
		
	def get_site_features(self):
		return self._site_features
		
	site_features = property(get_site_features,set_site_features,doc="Dict of site-level oxidation-state-independent properties")
		
	def ox_combos(self,site):
		"""
		Get info for combinations of oxidation states at a site
		For use in set_site_ox_features and set_comp_ox_features
		
		Parameters:
		-----------
		site: 'A','B','X', or 'comp' (for full composition)
		
		Returns:
		--------
		fixed_dict: dict of fixed-valence ions and their oxidation states, coordination numbers, and radii {el:{'r':r,'OS':OS,'CN':CN}}
		multi_dict: dict of multivalent ions, their oxidation states and coordination numbers, and corresponding radii {el:{n:{'r':r,'OS':OS,'CN':CN}}}
		multi_combos: list of of lists of oxidation states for each multivalent ion [[el1 ox states],[el2 ox states],...]
		"""			
		try:
			composition = self.site_composition[site]
			if self.silent==False:
				if site=='comp':
					print_site = "Full composition"
				else:
					print_site = f"{site}-site"
				print(f"{print_site} oxidation state combinations")
		except KeyError:
			raise Exception(f'Site {site} not recognized')
				
		#fixed-valence ions
		fixed = [el for el in composition.elements if len(self.allowed_ox_states[el.name]) == 1]
		fixed_dict = {}
		for el in fixed:
			ox = self.allowed_ox_states[el.name][0]
			#get closest CN to CN for site
			cn = self.site_CN[self.ion_site[el.name]]
			rom = self._closest_CN(el,ox,cn)
			if self.silent==False: 
				print('Fixed valence:',el.name,ox,rom)
			#get radius for CN (if 2 equally close CNs, average radii) - should only be one spin state
			radius = np.mean([el.data['Shannon radii'][f'{ox}'][rn][''][self.radius_type] for rn in rom])
			fixed_dict[el] = {'r':radius,'OS':ox, 'CN':rom}
		#multivalent ions
		multival = [el for el in composition.elements if el in self.multivalent_ions]
		multi_combos = []
		multi_dict = {}
		for el in multival:
			multi_combos.append(self.allowed_ox_states[el.name])
			md = {}
			#get Shannon radius for each ox state
			for ox in self.allowed_ox_states[el.name]:
				#get closest CN to CN for site
				cn = self.site_CN[self.ion_site[el.name]]
				rom = self._closest_CN(el,ox,cn)
				#get radius for CN (if 2 equally close CNs, average radii)
				radii = np.empty(len(rom))
				for i,rn in enumerate(rom):
					#assume transition metal assumes high-spin state if available (assumption made by Bartel)
					try:
						radii[i] = el.data['Shannon radii'][f'{ox}'][rn]['High Spin'][self.radius_type]
					except KeyError:
						radii[i] = el.data['Shannon radii'][f'{ox}'][rn][''][self.radius_type]
				radius = np.mean(radii)
				md[ox] = {'r':radius,'OS':ox, 'CN':rom}
			if self.silent==False:
				ox_CN = [f"{k} " + str(v['CN']) for k,v in md.items()]
				print('Multivalent:', el.name, ox_CN)
			multi_dict[el] = md
			
		return fixed_dict, multi_dict, multi_combos
	
	def site_ox_feature_labels(self,site):
		"""Labels for oxidation state-dependent site features"""
		features = ['r_mean','r_std','OS_mean','OS_std','tot_charge']
		if site in ['A','B','comp']:
			features += ['X_cat_mean','X_cat_std','ion_energy_mean','ion_energy_std']
		return features
		
	def site_ox_feature_categories(self,site):
		"""Categories for oxidation state-dependent site features"""
		categories = ['structure','structure','charge','charge','charge']
		if site in ['A','B','comp']:
			categories += ['bonding','bonding','charge','charge']
		return categories
	
	def site_ox_feature_units(self,site):
		"""Units for oxidation state-dependent site features"""
		units = ['length','length','charge','charge','charge']
		if site in ['A','B','comp']:
			units += ['charge/length2','charge/length2','energy','energy']
		return units
	
	def set_site_ox_features(self,site):
		"""
		Considers all possible combinations of oxidation states for multivalent ions on site
		Calculate site-level oxidation state-dependent features for site for each combination
		
		Parameters:
		-----------
		site: 'A','B', 'X', or 'comp' (for full composition)
		"""
		try:
			composition = self.site_composition[site]
		except KeyError:
			raise Exception(f'Site {site} not recognized')
		
		fixed_dict, multi_dict, multi_combos = self.ox_combos(site)
		
		#get weights before loop - don't depend on oxidation state
		weights = list(composition.get_el_amt_dict().values())
		#cation weights for 
		if site=='comp':
			cations = [el for el in composition.elements if el.name in self.cations]
			cat_weights = [composition.get(el) for el in cations]
		elif site in ['A','B']:
			cations = composition.elements
			cat_weights = weights
		
		feature_df = pd.DataFrame(columns=self.site_ox_feature_labels(site))
		for tup in itertools.product(*multi_combos): #get all combinations of oxidation states for multivalents
			ion_dict = fixed_dict.copy()
			for m, ox in zip(multi_dict.keys(),tup):
				#print(m,ox)
				ion_dict.update({m:multi_dict[m][ox]})
			#print(cat_dict)
			#get site average and standard deviation of radii and oxidation states
			
			
			r_mean = np.average([ion_dict[el]['r'] for el in composition.elements],weights=weights)
			r_std = np.average([(ion_dict[el]['r'] - r_mean)**2 for el in composition.elements],weights=weights)**0.5
			OS_mean = np.average([ion_dict[el]['OS'] for el in composition.elements],weights=weights)
			OS_std = np.average([(ion_dict[el]['OS'] - OS_mean)**2 for el in composition.elements],weights=weights)**0.5
			
			#total site charge
			tot_charge = OS_mean*sum(weights)
			
			features = [r_mean,r_std,OS_mean,OS_std,tot_charge]
			
			#cation-only features: cation electronegativity and ionization energy - calculate for A and B sites, and all cations for comp
			if site in ['A','B','comp']:
				# try:
					# MX_ABE = self.site_MX_ABE(site, {el.name:state['OS'] for el,state in ion_dict.items()})
				# except LookupError:
					# #if there is no thermo data for one or more of the binary compounds corresponding to these oxidation states, return NaN
					# #NaNs will be excluded when aggregating across oxidation state combinations
					# MX_ABE = np.nan
				
				#"cation electronegativity" (as defined in Zohourian 2018: z/r^2)
				X_cat_mean = np.average([ion_dict[el]['OS']/ion_dict[el]['r']**2 for el in cations],weights=cat_weights)
				X_cat_std = np.average([(ion_dict[el]['OS']/ion_dict[el]['r']**2 - X_cat_mean)**2 for el in cations],weights=cat_weights)**0.5
					
				#ionization energy
				def ion_energy(el,ion_dict):
					return sum([energy for num,energy in mdl.element(el.name).ionenergies.items() if num <= ion_dict[el]['OS']])
				
				ion_energy_mean = np.average([ion_energy(el,ion_dict) for el in cations],weights=cat_weights)
				#self.site_mean_func(site,lambda el: ion_energy(el,ion_dict))
				ion_energy_std = np.average([(ion_energy(el,ion_dict) - ion_energy_mean)**2 for el in cations],weights=cat_weights)**0.5
				#self.site_mean_func(site,lambda el: (ion_energy(el,ion_dict) - ion_energy_mean)**2)**0.5
				
				features += [X_cat_mean,X_cat_std,ion_energy_mean,ion_energy_std]
			
			#put all features for current ox combo into row of DataFrame 
			idx_list = ["{}{}".format(el.name,state['OS']) for el,state in ion_dict.items()]
			df_index = '_'.join(idx_list)
			
			feature_df = feature_df.append(pd.Series(features,name=df_index,index=self.site_ox_feature_labels(site)))
		
		#print('called set_ox_combos') #used to verify that set_ox_combos runs only once initially
		self._site_ox_features[site] = feature_df
	
	def get_site_ox_features(self):
		return self._site_ox_features
	
	site_ox_features = property(get_site_ox_features,set_site_ox_features,doc="DataFrame of site-level oxidation state-dependent properties")

	@property
	def comp_ox_feature_labels(self):
		"""
		List of composition-level oxidation state-dependent properties
		"""
		return ['goldschmidt','goldschmidt_struct','tau','tot_cat_charge','anion_delta','alat_hardsphere','uc_vol','uc_vol_free','r_crit','r_AB_ratio']
		
	@property
	def comp_ox_feature_categories(self):
		"""
		List of composition-level oxidation state-dependent categories
		"""
		return ['structure','structure','structure','charge','charge','structure','structure','structure','structure','structure']
		
	@property
	def comp_ox_feature_units(self):
		"""
		List of composition-level oxidation state-dependent units
		"""
		return ['none','none','none','charge','none','length','volume','volume','length','none']
	
	def set_comp_ox_features(self):
		"""
		Considers all possible combinations of oxidation states for multivalent ions in full composition
		Calculates composition-level oxidation state-dependent properties for each combination
		"""
		
		fixed_dict, multi_dict, multi_combos = self.ox_combos('comp')
		composition = self.composition
		
		#get weights before loop - don't depend on oxidation state
		weights = list(composition.get_el_amt_dict().values())
		cations = [el for el in composition.elements if el.name in self.cations]
		cat_weights = [composition.get(el) for el in cations]
		
		feature_df = pd.DataFrame(columns=self.comp_ox_feature_labels)
		for tup in itertools.product(*multi_combos): #get all combinations of oxidation states for multivalents
			ion_dict = fixed_dict.copy()
			for m, ox in zip(multi_dict.keys(),tup):
				ion_dict.update({m:multi_dict[m][ox]})
			
			# #get site average and standard deviation of radii and oxidation states
			# r_mean = np.average([ion_dict[el]['r'] for el in composition.elements],weights=weights)
			# r_std = np.average([(ion_dict[el]['r'] - r_mean)**2 for el in composition.elements],weights=weights)**0.5
			# OS_mean = np.average([ion_dict[el]['OS'] for el in composition.elements],weights=weights)
			# OS_std = np.average([(ion_dict[el]['OS'] - OS_mean)**2 for el in composition.elements],weights=weights)**0.5
			
			# #"cation electronegativity" (as defined in Zohourian 2018: z/r^2) - should only include cations, not anion
			# X_cat_mean = np.average([ion_dict[el]['OS']/ion_dict[el]['r']**2 for el in cations],weights=cat_weights)
			# X_cat_std = np.average([(ion_dict[el]['OS']/ion_dict[el]['r']**2 - X_cat_mean)**2 for el in cations],weights=cat_weights)**0.5
			
			#look up A, B, and X site info for current ox combo
			site_radii = np.empty(3)
			site_ox_states = np.empty(3)
			#site_ABE = np.empty(2)
			for i,site in enumerate('ABX'):
				if site not in self.site_ox_features.keys():
					#calculate ox features for site if not existing
					self.set_site_ox_features(site)
				#look up features for oxidation state combo in dataframe	
				idx_list = ["{}{}".format(el.name,state['OS']) for el,state in ion_dict.items() if self.ion_site[el.name]==site]
				index = '_'.join(idx_list)
				site_radii[i] = self.site_ox_features[site].loc[index,'r_mean']
				site_ox_states[i] = self.site_ox_features[site].loc[index,'OS_mean']
				#if site in 'AB':
					#site_ABE[i] = self.site_ox_features[site].loc[index,'MX_ABE']
			
			ra, rb, rx = site_radii
			na, nb, nx = site_ox_states
			#BEa, BEb = site_ABE
			
			#tolerance factors
			goldschmidt = (ra+rx)/((2**0.5)*(rb+rx))
			tau = (rx/rb)-na*(na-(ra/rb)/np.log(ra/rb)) #New tolerance factor from Bartel et al. (2019), Sci. Adv. 5(2).
			
			#goldschmidt predicted structure
			if goldschmidt < 0.71:
				goldschmidt_struct = 0 #other structure
			elif goldschmidt < 0.9:
				goldschmidt_struct = 1 #orthorhombic/rhombohedral
			elif goldschmidt <= 1:
				goldschmidt_struct = 2 #cubic
			else:
				goldschmidt_struct = 3 #hexagonal/tetragonal
			
			#total cation charge & anion delta
			tot_cat_charge = na*self.site_sum('A') + nb*self.site_sum('B')
			anion_delta = self.nominal_site_amt['X'] + (tot_cat_charge/self.site_base_ox['X']) #anion non-stoichiometry
			
			#unit cell volume and free volume (assume cubic)
			if goldschmidt > 1: #A-O bonds are close packed
				alat_hardsphere = (2**0.5)*(ra + rx)
			elif goldschmidt <= 1: #B-O bonds are close packed
				alat_hardsphere = 2*(rb + rx)
			uc_vol = alat_hardsphere**3
			uc_vol_free = uc_vol - (4*np.pi/3)*(ra**3+rb**3+(3-anion_delta)*rx**3)
			
			#critical radius of saddle point. From Liu et al. (2011), J. Memb. Sci. 383, 235-240.
			r_crit = (-ra**2 + (3/4)*alat_hardsphere**2 - (2**0.5)*alat_hardsphere*rb + rb**2) / (2*ra + (2**0.5)*alat_hardsphere - 2*rb)
			
			r_AB_ratio = ra/rb
			
			##metal-anion average bond energy per mole of metal. From Sammells et al. (1992), Solid State Ionics 52, 111-123. (Modified to weight for site occupancies)
			#MX_ABE = 2*np.average([BEa/12, BEb/6],weights=[self.site_sum('A'),self.site_sum('B')])
			
			#put all features for current ox combo into row of DataFrame 
			idx_list = ["{}{}".format(el.name,state['OS']) for el,state in ion_dict.items()]
			df_index = '_'.join(idx_list)
			feature_df = feature_df.append(pd.Series([goldschmidt,goldschmidt_struct,tau,tot_cat_charge,anion_delta,alat_hardsphere,uc_vol,uc_vol_free,r_crit,r_AB_ratio],
						name=df_index, index=self.comp_ox_feature_labels))
				
		self._comp_ox_features = feature_df
		
	def get_comp_ox_features(self):
		return self._comp_ox_features
	
	comp_ox_features = property(get_comp_ox_features,set_comp_ox_features,doc="DataFrame of composition-level oxidation state-dependent properties")
	
	def featurize(self,sites='ABX',ox_stats=['min','max','mean','median','std','range']):
		"""
		Generate full feature set for perovskite. Includes composition-level features, and site-level features as specified.
		Oxidation-state-dependent properties are aggregated into features using the functions specified in ox_stats
		
		Parameters:
		-----------
		sites: list or string of sites to featurize. Any combination of 'A', 'B', 'X', and/or 'comp' accepted. 
			Passing '' or [] will return only composition-level oxidation-state-dependent features.
			Including 'comp' will calculate oxidation-state-independent features for the full composition
		ox_stats: list of aggregate functions to apply to oxidation state combinations for feature generation. 
			Options: 'min','max','mean','median','std','range', or any built-in pandas aggregate function
		
		Returns: pandas Series of feature values
		"""
		
		#calculate site-level oxidation-state-dependent features for all sites - required for composition-level features
		for site in 'ABX':
			self.set_site_ox_features(site)
		self.set_comp_ox_features()
		
		features = pd.Series()
		
		#aggregate composition-level oxidation-state-dependent properties into features
		for stat in ox_stats:
			if stat=='range':
				stat_feat = (self.comp_ox_features.max() - self.comp_ox_features.min()).add_prefix(f'comp_ox{stat}_')
			elif stat=='std':
				stat_feat = getattr(self.comp_ox_features,stat)().add_prefix(f'comp_ox{stat}_').fillna(0)
			else:
				stat_feat = getattr(self.comp_ox_features,stat)().add_prefix(f'comp_ox{stat}_')
			features = pd.concat([features,stat_feat])
			
		#aggregate site-level oxidation-state-dependent properties into features
		for site in sites:
			if site=='comp':
				self.set_site_ox_features('comp')
				site_label = 'comp'
			else:
				site_label = f'{site}site'
			for stat in ox_stats:
				if stat=='range':
					#no built-in range function
					stat_feat = (self.site_ox_features[site].max() - self.site_ox_features[site].min()).add_prefix(f'{site_label}_ox{stat}_')
				elif stat=='std':
					#if only one value, std() returns NaN. Convert to 0
					stat_feat = getattr(self.site_ox_features[site],stat)().add_prefix(f'{site_label}_ox{stat}_').fillna(0)
				else:
					stat_feat = getattr(self.site_ox_features[site],stat)().add_prefix(f'{site_label}_ox{stat}_')
				features = pd.concat([features,stat_feat])
		
		#add exp thermo features
		thermo_feat = pd.Series(self.get_exp_thermo_features(sites,ox_stats))
		features = pd.concat([features,thermo_feat])
		
		#add oxidation-state-independent features
		for site in sites:
			if site=='comp':
				site_label = 'comp'
			else:
				site_label = f'{site}site'
			self.set_site_features(site)
			features = pd.concat([features,pd.Series(self.site_features[site]).add_prefix(f'{site_label}_')])
			
		#add A:B site occupancy ratio
		AB_ratio = self.site_sum('A')/self.site_sum('B')
		features = pd.concat([features,pd.Series({'AB_ratio':AB_ratio})])
		
		return features
		
	def feature_labels(self,sites='ABX',ox_stats=['min','max','mean','median','std','range']):
		"""
		Get list of feature labels
		
		Parameters:
		-----------
		sites: sites to featurize. Any combination of A, B, and/or X accepted. Passing '' or [] will return only composition-level features
		ox_stats: list of aggregate functions to apply to oxidation state combinations for feature generation. Options: 'min','max','mean','median','std','range'
		
		Returns: list of feature labels (strings)
		"""
		feature_labels = []
		#composition-level oxidation-state-aggregate features
		for stat in ox_stats:
			stat_feat = [f'comp_ox{stat}_{label}' for label in self.comp_ox_feature_labels]
			feature_labels += stat_feat
		
		#site-level oxidation-state-aggregate features
		for site in sites:
			if site=='comp':
				site_label = 'comp'
			else:
				site_label = f'{site}site'
			for stat in ox_stats:
				stat_feat = [f'{site_label}_ox{stat}_{label}' for label in self.site_ox_feature_labels(site)]
				feature_labels += stat_feat
			
		#MX_ABE features
		feature_labels += list(self.exp_thermo_feature_labels(sites,ox_stats))
		
		#site-level oxidation-state-independent features
		for site in sites:
			if site=='comp':
				site_label = 'comp'
			else:
				site_label = f'{site}site'
			site_feat = [f'{site_label}_{label}' for label in self.site_feature_labels(site)]
			feature_labels += site_feat
		
		#A:B ratio	 
		feature_labels += ['AB_ratio']
			
		return feature_labels
		
	def feature_categories(self,sites='ABX',ox_stats=['min','max','mean','median','std','range']):
		"""
		Get list of feature categories
		
		Parameters:
		-----------
		sites: sites to featurize. Any combination of A, B, and/or X accepted. Passing '' or [] will return only composition-level features
		ox_stats: list of aggregate functions to apply to oxidation state combinations for feature generation. Options: 'min','max','mean','median','std','range'
		
		Returns: list of feature labels (strings)
		"""
		feature_categories = []
		#composition-level oxidation-state-aggregate features
		for stat in ox_stats:
			feature_categories += self.comp_ox_feature_categories
		
		#site-level oxidation-state-aggregate features
		for site in sites:
			for stat in ox_stats:
				feature_categories += self.site_ox_feature_categories(site)
			
		#MX_ABE features
		feature_categories += list(self.exp_thermo_feature_categories(sites,ox_stats))
		
		#site-level oxidation-state-independent features
		for site in sites:
			feature_categories += self.site_feature_categories(site)
		
		#A:B ratio	 
		feature_categories += ['composition']
			
		return feature_categories	
		
	def feature_units(self,sites='ABX',ox_stats=['min','max','mean','median','std','range']):
		"""
		Get list of feature units
		
		Parameters:
		-----------
		sites: sites to featurize. Any combination of A, B, and/or X accepted. Passing '' or [] will return only composition-level features
		ox_stats: list of aggregate functions to apply to oxidation state combinations for feature generation. Options: 'min','max','mean','median','std','range'
		
		Returns: list of feature labels (strings)
		"""
		feature_units = []
		#composition-level oxidation-state-aggregate features
		for stat in ox_stats:
			feature_units += self.comp_ox_feature_units
		
		#site-level oxidation-state-aggregate features
		for site in sites:
			for stat in ox_stats:
				feature_units += self.site_ox_feature_units(site)
			
		#MX_ABE features
		feature_units += list(self.exp_thermo_feature_units(sites,ox_stats))
		
		#site-level oxidation-state-independent features
		for site in sites:
			feature_units += self.site_feature_units(site)
		
		#A:B ratio	 
		feature_units += ['none']
			
		return feature_units
	
def formula_redfeat(formula,cat_ox_lims={}):
	pvskt = perovskite(formula,site_ox_lim={'A':[2,4],'B':[2,4]},site_base_ox={'A':2,'B':4})
	for k,v in cat_ox_lims.items():
		pvskt.set_cat_ox_lim(k,v)
	pvskt.featurize()
	red_feat = {'{}'.format(k):v for (k,v) in pvskt.features.items() 
				if k[-5:] not in ['oxmin','oxmax'] and k[0:7]!='O_delta'}
	return red_feat
	
def formula_pif(formula,cat_ox_lims={},red_feat=None):
	'''
	create pif with formula and chemical feature properties
	'''
	fpif = ChemicalSystem()
	fpif.chemical_formula = formula
	if red_feat is None:
		red_feat = formula_readfeat(formula,cat_ox_lims)
	
	props = []
	for feat, val in red_feat.items():
		prop = Property(name=feat,scalars=val)
		props.append(prop)
	fpif.properties=props
	
	return fpif, red_feat
			
	
