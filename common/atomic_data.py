# Copyright (C) 2008 NSC Jyvaskyla
# Please see the accompanying LICENSE file for further information.
#
# Experimental data * mass, R_cov (2008 data), R_vdw, EA from www.webelements.com (updated 21/May/2015)
#                   * IE from gElemental 1.2.0
#                   * EN according to Allred and Rochow (Wiley-VCH periodic table, 2007)
#
# UNITS:
#     * mass in amu
#     * all radii in Angstrom
#     * all energies in eV

from numpy import nan

atomic_data={}
atomic_data['H'] ={'Z':1, 'symbol':'H',  'name':'hydrogen',  'mass': 1.0079, 'R_cov':0.31, 'R_vdw':1.20, 'IE':0.0135, 'EA':72.27, 'add_orb':'1s', 'rem_orb':'1s', 'EN':2.2 }            
atomic_data['He']={'Z':2, 'symbol':'He', 'name':'helium',                                                                         'add_orb':'2s', 'rem_orb':'1s' }
atomic_data['Li']={'Z':3, 'symbol':'Li', 'name':'lithium',   'mass':6.941,   'R_cov':1.28, 'R_vdw':1.82,                          'add_orb':'2s', 'rem_orb':'2s', 'EN':1.  }
atomic_data['Be']={'Z':4, 'symbol':'Be', 'name':'beryllium', 'mass':9.0122,  'R_cov':0.96,                                        'add_orb':'2p', 'rem_orb':'2s', 'EN':1.5 }
atomic_data['B'] ={'Z':5, 'symbol':'B',  'name':'boron',     'mass':10.81,   'R_cov':0.84, 'R_vdw':2.08, 'IE':8.294,  'EA':0.277, 'add_orb':'2p', 'rem_orb':'2p', 'EN':2.  }
atomic_data['C'] ={'Z':6, 'symbol':'C',  'name':'carbon',    'mass':12.0107, 'R_cov':0.76, 'R_vdw':1.70, 'IE':11.256, 'EA':1.594, 'add_orb':'2p', 'rem_orb':'2p', 'EN':2.5 }            
atomic_data['N'] ={'Z':7, 'symbol':'N',  'name':'nitrogen',  'mass':14.0067, 'R_cov':0.71, 'R_vdw':1.55, 'IE':14.527, 'EA':0.072, 'add_orb':'2p', 'rem_orb':'2p', 'EN':3.1 }            
atomic_data['O'] ={'Z':8, 'symbol':'O',  'name':'oxygen',    'mass':15.9994, 'R_cov':0.66, 'R_vdw':1.52, 'IE':13.612, 'EA':1.460, 'add_orb':'2p', 'rem_orb':'2p', 'EN':3.5 }            
atomic_data['F'] ={'Z':9, 'symbol':'F',  'name':'fluorine',  'mass':18.9984, 'R_cov':0.57, 'R_vdw':1.47, 'IE':17.4228,'EA':3.4012,'add_orb':'2p', 'rem_orb':'2p', 'EN':4.1 }    
atomic_data['Ne']={'Z':10,'symbol':'Ne', 'name':'neon',                                                                           'add_orb':'3s', 'rem_orb':'2p' }
atomic_data['Na']={'Z':11,'symbol':'Na', 'name':'sodium',    'mass':22.9898, 'R_cov':1.66, 'R_vdw':2.27, 'IE':5.136,  'EA':0.547, 'add_orb':'3s', 'rem_orb':'3s', 'EN':1.  }           
atomic_data['Mg']={'Z':12,'symbol':'Mg', 'name':'magnesium', 'mass':24.3050, 'R_cov':1.41, 'R_vdw':1.73, 'IE':7.642,  'EA':0.000, 'add_orb':'3p', 'rem_orb':'3s', 'EN':1.2 }             
atomic_data['Al']={'Z':13,'symbol':'Al', 'name':'aluminium', 'mass':26.9815, 'R_cov':1.21, 'R_vdw':nan,  'IE':5.986,              'add_orb':'3p', 'rem_orb':'3p', 'EN':1.5 }
atomic_data['Si']={'Z':14,'symbol':'Si', 'name':'silicon',   'mass':28.0855, 'R_cov':1.11, 'R_vdw':2.10, 'IE':8.151,              'add_orb':'3p', 'rem_orb':'3p', 'EN':1.7 }
atomic_data['P'] ={'Z':15,'symbol':'P',  'name':'phosphorus','mass':30.9738, 'R_cov':1.07, 'R_vdw':1.80, 'IE':10.486,             'add_orb':'3p', 'rem_orb':'3p', 'EN':2.1 }
atomic_data['S'] ={'Z':16,'symbol':'S',  'name':'sulfur',    'mass':32.065,  'R_cov':1.05, 'R_vdw':1.80, 'IE':10.356, 'EA':2.072, 'add_orb':'3p', 'rem_orb':'3p', 'EN':2.4 }
atomic_data['Cl']={'Z':17,'symbol':'Cl', 'name':'chlorine',  'mass':35.4530, 'R_cov':1.02, 'R_vdw':1.75, 'IE':12.962, 'EA':3.615, 'add_orb':'3p', 'rem_orb':'3p', 'EN':2.8 }            
atomic_data['Ar']={'Z':18,'symbol':'Ar', 'name':'argon',     'mass':39.948,                                                       'add_orb':'3p', 'rem_orb':'4s' }
atomic_data['K'] ={'Z':19,'symbol':'K',  'name':'potassium', 'mass':39.0983, 'R_cov':2.03, 'R_vdw':2.75, 'IE':4.338,  'EA':0.501, 'add_orb':'4s', 'rem_orb':'4s', 'EN':0.9 }             
atomic_data['Ca']={'Z':20,'symbol':'Ca', 'name':'calcium',   'mass':40.078,  'R_cov':1.41,               'IE':6.113,              'add_orb':'3d', 'rem_orb':'4s', 'EN':1.  }
atomic_data['Sc']={'Z':21,'symbol':'Sc', 'name':'scandium',  'mass':44.9559, 'R_cov':1.44,               'IE':6.54,               'add_orb':'3d', 'rem_orb':'3d', 'EN':1.2 }
atomic_data['Ti']={'Z':22,'symbol':'Ti', 'name':'titanium',  'mass':47.8760, 'R_cov':1.60, 'R_vdw':2.15, 'IE':6.825,  'EA':0.078, 'add_orb':'3d', 'rem_orb':'3d', 'EN':1.3 }            
atomic_data['V'] ={'Z':23,'symbol':'V',  'name':'vanadium',  'mass':50.942,  'R_cov':1.22, 'add_orb':'3d', 'rem_orb':'3d'}
atomic_data['Cr']={'Z':24,'symbol':'Cr', 'name':'chromium',  'mass':51.9961, 'R_cov':1.39, 'R_vdw':nan,  'IE':6.766,              'add_orb':'3d', 'rem_orb':'3d', 'EN':1.6 }
atomic_data['Mn']={'Z':25,'symbol':'Mn', 'name':'manganese', 'mass':54.938,  'R_cov':1.17, 'add_orb':'3d', 'rem_orb':'3d'}
atomic_data['Fe']={'Z':26,'symbol':'Fe', 'name':'iron',      'mass':55.845,  'R_cov':1.32, 'R_cov_hs':1.52, 'IE':7.870,        'add_orb':'3d', 'rem_orb':'3d', 'EN':1.6 }
atomic_data['Co']={'Z':27,'symbol':'Co', 'name':'cobalt',    'mass':58.933,  'R_cov':1.16, 'add_orb':'3d', 'rem_orb':'3d'}
atomic_data['Ni']={'Z':28,'symbol':'Ni', 'name':'nickel',    'mass':58.6934, 'R_cov':1.24, 'R_vdw':1.63, 'IE':7.635,              'add_orb':'3d', 'rem_orb':'3d', 'EN':1.5 }
atomic_data['Cu']={'Z':29,'symbol':'Cu', 'name':'copper',    'mass':63.546,  'R_cov':1.38, 'R_vdw':2.00, 'IE':7.727,  'EA':1.227, 'add_orb':'4s', 'rem_orb':'4s', 'EN':1.8 }
atomic_data['Zn']={'Z':30,'symbol':'Zn', 'name':'zinc',      'mass':65.38,   'R_cov':1.25, 'add_orb':'4p', 'rem_orb':'4s'}
atomic_data['Ga']={'Z':31,'symbol':'Ga', 'name':'gallium',   'mass':69.723,  'R_cov':1.26, 'add_orb':'4p', 'rem_orb':'4p'}
atomic_data['Ge']={'Z':32,'symbol':'Ge', 'name':'germanium', 'mass':62.631,  'R_cov':1.22, 'add_orb':'4p', 'rem_orb':'4p'}
atomic_data['As']={'Z':33,'symbol':'As', 'name':'arsenic',   'mass':74.922,  'R_cov':1.20, 'add_orb':'4p', 'rem_orb':'4p'}
atomic_data['Se']={'Z':34,'symbol':'Se', 'name':'selenium',  'mass':78.971,  'R_cov':1.16, 'add_orb':'4p', 'rem_orb':'4p'}
atomic_data['Br']={'Z':35,'symbol':'Br', 'name':'bromine',   'mass':79.904,  'R_cov':1.20, 'R_vdw':1.85, 'IE':11.814,             'add_orb':'4p', 'rem_orb':'4p', 'EN':2.7 }
atomic_data['Kr']={'Z':36,'symbol':'Kr', 'name':'krypton',                                                                        'add_orb':'5s', 'rem_orb':'4p' }
atomic_data['Rb']={'Z':37,'symbol':'Rb', 'name':'rubidium',  'mass':84.468,  'R_cov':2.16, 'add_orb':'5s', 'rem_orb':'5s'}
atomic_data['Sr']={'Z':38,'symbol':'Sr', 'name':'strontium', 'mass':87.62,   'R_cov':1.95, 'R_vdw':2.49, 'IE':5.69,   'EA':0.052, 'add_orb':'4d', 'rem_orb':'5s', 'EN':1.  }
atomic_data['Y']= {'Z':39,'symbol':'Y',  'name':'yttrium',   'mass':88.906,  'R_cov':1.62, 'add_orb':'4d', 'rem_orb':'4d'}
atomic_data['Zr']={'Z':40,'symbol':'Zr', 'name':'zirconium', 'mass':91.224,  'R_cov':1.45, 'add_orb':'4d', 'rem_orb':'4d'}
atomic_data['Nb']={'Z':41,'symbol':'Nb', 'name':'niobium',   'mass':92.906,  'R_cov':1.34, 'add_orb':'4d', 'rem_orb':'4d'}
atomic_data['Mo']={'Z':42,'symbol':'Mo', 'name':'molybdenum','mass':95.94,   'R_cov':1.54, 'R_vdw':2.10, 'IE':7.08,   'EA':0.744, 'add_orb':'5s', 'rem_orb':'4d', 'EN':1.3 }
atomic_data['Tc']={'Z':43,'symbol':'Tc', 'name':'technetium','mass':98.907,  'R_cov':1.27, 'add_orb':'4d', 'rem_orb':'5s'}
atomic_data['Ru']={'Z':44,'symbol':'Ru', 'name':'ruthenium', 'mass':101.07,  'R_cov':1.46,               'IE':7.37,               'add_orb':'4d', 'rem_orb':'4d', 'EN':1.4 }
atomic_data['Rh']={'Z':45,'symbol':'Rh', 'name':'rhodium',   'mass':102.9055,'R_cov':1.42,               'IE':7.46,               'add_orb':'4d', 'rem_orb':'4d', 'EN':1.5 }
atomic_data['Pd']={'Z':46,'symbol':'Pd', 'name':'palladium', 'mass':106.42,  'R_cov':1.39, 'R_vdw':1.63, 'IE':8.337,              'add_orb':'5s', 'rem_orb':'4d', 'EN':1.4 }
atomic_data['Ag']={'Z':47,'symbol':'Ag', 'name':'silver',    'mass':107.868, 'R_cov':1.45, 'R_vdw':1.72, 'IE':7.576,  'EA':1.302, 'add_orb':'5s', 'rem_orb':'5s', 'EN':1.4 }
atomic_data['Cd']={'Z':48,'symbol':'Cd', 'name':'cadmium',   'mass':112.414, 'R_cov':1.48, 'add_orb':'5p', 'rem_orb':'5s'}
atomic_data['In']={'Z':49,'symbol':'In', 'name':'indium',    'mass':114.818, 'R_cov':1.44, 'add_orb':'5p', 'rem_orb':'5p'}
atomic_data['Sn']={'Z':50,'symbol':'Sn', 'name':'tin',       'mass':118.710, 'R_cov':1.39, 'R_vdw':2.17, 'IE':7.344,              'add_orb':'5p', 'rem_orb':'5p', 'EN':1.7 }
atomic_data['Sb']={'Z':51,'symbol':'Sb', 'name':'antimony',  'mass':121.760, 'R_cov':1.40, 'add_orb':'5p', 'rem_orb':'5p'}
atomic_data['Te']={'Z':52,'symbol':'Te', 'name':'tellurium', 'mass':127.6,   'R_cov':1.36, 'add_orb':'5p', 'rem_orb':'5p'}
atomic_data['I'] ={'Z':53,'symbol':'I',  'name':'iodine',    'mass':126.9045,'R_cov':1.33, 'R_vdw':2.20, 'IE':10.451,             'add_orb':'5p', 'rem_orb':'5p', 'EN':2.2 }
atomic_data['Xe']={'Z':54,'symbol':'Xe', 'name':'xenon',     'mass':131.293,               'R_vdw':2.10, 'IE':12.130,             'add_orb':'6s', 'rem_orb':'5p' }
atomic_data['Cs']={'Z':55,'symbol':'Cs', 'name':'cesium',    'mass':132.905, 'R_cov':2.35, 'add_orb':'6s', 'rem_orb':'6s'}
atomic_data['Ba']={'Z':56,'symbol':'Ba', 'name':'barium',    'mass':137.328, 'R_cov':1.98, 'add_orb':'5d', 'rem_orb':'6s'}
atomic_data['Lu']={'Z':71,'symbol':'Lu', 'name':'lutetium',  'mass':174.967, 'R_cov':1.56, 'add_orb':'5d', 'rem_orb':'5d'}
atomic_data['Hf']={'Z':72,'symbol':'Hf', 'name':'hafnium',   'mass':178.49,  'R_cov':1.44, 'add_orb':'5d', 'rem_orb':'5d'}
atomic_data['Ta']={'Z':73,'symbol':'Ta', 'name':'tantalum',  'mass':180.948, 'R_cov':1.34, 'add_orb':'5d', 'rem_orb':'5d'}
atomic_data['W'] ={'Z':74,'symbol':'W',  'name':'tungsten',  'mass':183.84,  'R_cov':1.30, 'add_orb':'5d', 'rem_orb':'5d'}
atomic_data['Re']={'Z':75,'symbol':'Re', 'name':'rhenium',   'mass':186.207, 'R_cov':1.28, 'add_orb':'5d', 'rem_orb':'5d'}
atomic_data['Os']={'Z':76,'symbol':'Os', 'name':'osmium',    'mass':190.23,  'R_cov':1.26, 'add_orb':'5d', 'rem_orb':'5d'}
atomic_data['Ir']={'Z':77,'symbol':'Ir', 'name':'iridium',   'mass':192.217, 'R_cov':1.27, 'add_orb':'5d', 'rem_orb':'5d'}
atomic_data['Pt']={'Z':78,'symbol':'Pt', 'name':'platinum',  'mass':195.084, 'R_cov':1.36,'R_vdw':1.75, 'IE':9.013,  'EA':2.127, 'add_orb':'5d', 'rem_orb':'6s', 'EN':1.4 }
atomic_data['Au']={'Z':79,'symbol':'Au', 'name':'gold',      'mass':196.9666,'R_cov':1.36,'R_vdw':1.66, 'IE':9.221,  'EA':2.308, 'add_orb':'6s', 'rem_orb':'6s', 'EN':1.4 }
atomic_data['Hg']={'Z':80,'symbol':'Hg', 'name':'mercury',   'mass':200.592, 'R_cov':1.49, 'add_orb':'6s', 'rem_orb':'6p'}
atomic_data['Tl']={'Z':81,'symbol':'Tl', 'name':'thallium',  'mass':204.383, 'R_cov':1.48, 'add_orb':'6p', 'rem_orb':'6p'}
atomic_data['Pb']={'Z':82,'symbol':'Pb', 'name':'lead',      'mass':207.2,   'R_cov':1.47, 'add_orb':'6p', 'rem_orb':'6p'}
atomic_data['Bi']={'Z':83,'symbol':'Bi', 'name':'bismuth',   'mass':208.980, 'R_cov':1.46, 'add_orb':'6p', 'rem_orb':'6p'}
atomic_data['Po']={'Z':84,'symbol':'Po', 'name':'polonium',  'mass':208.982, 'R_cov':1.46, 'add_orb':'6p', 'rem_orb':'6p'}
atomic_data['At']={'Z':85,'symbol':'At', 'name':'astatine',  'mass':209.987, 'R_cov':1.45, 'add_orb':'6p', 'rem_orb':'6p'}
atomic_data['Rn']={'Z':86,'symbol':'Rn', 'name':'radon',     'mass':222.018, 'R_cov':nan,  'add_orb':'7s', 'rem_orb':'6p'}
atomic_data['Fr']={'Z':87,'symbol':'Fr', 'name':'francium',  'mass':223.020, 'R_cov':nan,  'add_orb':'7s', 'rem_orb':'7s'}
atomic_data['Ra']={'Z':88,'symbol':'Ra', 'name':'radium',    'mass':226.025, 'R_cov':nan,  'add_orb':'7s', 'rem_orb':'6d'}
atomic_data['Ac']={'Z':89,'symbol':'Ac', 'name':'actinium',  'mass':227.0278,                            'IE':5.381,              'add_orb':'6d', 'rem_orb':'6d', 'EN':1.  }   
atomic_data['Th']={'Z':90,'symbol':'Th','name':'thorium',    'mass':232.0381, 'R_cov':1.65,              'IE':6.307,              'add_orb':'5f', 'rem_orb':'6d', 'EN':1.1 }     
atomic_data['U'] ={'Z':92,'symbol':'U', 'name':'uranium',    'mass':238.0289, 'R_cov':1.42,              'IE':6.194,              'add_orb':'5f', 'rem_orb':'5f', 'EN':1.2 }   
atomic_data['Np']={'Z':93,'symbol':'Np','name':'neptunium',  'mass':237.048,                             'IE':6.266,              'add_orb':'5f', 'rem_orb':'6d', 'EN':1.2 }
atomic_data['Pu']={'Z':94,'symbol':'Pu','name':'plutonium',  'mass':244.0642,                            'IE':6.026,              'add_orb':'5f', 'rem_orb':'5f', 'EN':1.2 }    
atomic_data['X'] ={'Z':99,'symbol':'X', 'name':'dummy'}
## 'add_orb' and 'rem_orb' for calculations of IP and EA
#  (e- to 'add_orb' => Anion, e- from 'rem_orb' => cation) 

# update with valence orbital data
valence_orbitals = {
    'H' : ['1s'],
    'He': ['1s'],
    'Li': ['2s', '2p'],
    'Be': ['2s', '2p'],
    'B' : ['2s', '2p'],
    'C' : ['2s', '2p'],
    'N' : ['2s', '2p'],
    'O' : ['2s', '2p'],
    'F' : ['2s', '2p'],
    'Ne': ['2s', '2p'],
    'Na': ['3s', '3p'],
    'Mg': ['3s', '3p'],
    'Al': ['3s', '3p'],
    'Si': ['3s', '3p'],
    'P' : ['3s', '3p'],
    'S' : ['3s', '3p'],
    'Cl': ['3s', '3p'],
    'Ar': ['3s', '3p'],
    'K' : ['3d', '4s', '4p'],
    'Ca': ['3d', '4s', '4p'],
    'Sc': ['3d', '4s', '4p'],
    'Ti': ['3d', '4s', '4p'],
    'V' : ['3d', '4s', '4p'],
    'Cr': ['3d', '4s', '4p'],
    'Mn': ['3d', '4s', '4p'],
    'Fe': ['3d', '4s', '4p'],
    'Co': ['3d', '4s', '4p'],
    'Ni': ['3d', '4s', '4p'],
    'Cu': ['3d', '4s', '4p'],
    'Zn': ['3d', '4s', '4p'],
    'Ga': ['3d', '4s', '4p'],
    'Ge': ['3d', '4s', '4p'],
    'As': ['3d', '4s', '4p'],
    'Se': ['3d', '4s', '4p'],
    'Br': ['3d', '4s', '4p'],
    'Kr': ['3d', '4s', '4p'],
    'Rb': ['4d', '5s', '5p'],
    'Sr': ['4d', '5s', '5p'],
    'Y' : ['4d', '5s', '5p'],
    'Zr': ['4d', '5s', '5p'],
    'Nb': ['4d', '5s', '5p'],
    'Mo': ['4d', '5s', '5p'],
    'Tc': ['4d', '5s', '5p'],
    'Ru': ['4d', '5s', '5p'],
    'Rh': ['4d', '5s', '5p'],
    'Pd': ['4d', '5s', '5p'],
    'Ag': ['4d', '5s', '5p'],
    'Cd': ['4d', '5s', '5p'],
    'In': ['4d', '5s', '5p'],
    'Sn': ['4d', '5s', '5p'],
    'Sb': ['4d', '5s', '5p'],
    'Te': ['4d', '5s', '5p'],
    'I' : ['4d', '5s', '5p'],
    'Xe': ['4d', '5s', '5p'],
    'Cs': ['5d', '6s', '5p'],
    'Ba': ['5d', '6s', '5p'],
    'Lu': ['5d', '6s', '6p'],
    'Hf': ['5d', '6s', '6p'],
    'Ta': ['5d', '6s', '6p'],
    'W' : ['5d', '6s', '6p'],
    'Re': ['5d', '6s', '6p'],
    'Os': ['5d', '6s', '6p'],
    'Ir': ['5d', '6s', '6p'],
    'Pt': ['5d', '6s', '6p'],
    'Au': ['5d', '6s', '6p'],
    'Hg': ['5d', '6s', '6p'],
    'Tl': ['5d', '6s', '6p'],
    'Pb': ['5d', '6s', '6p'],
    'Bi': ['5d', '6s', '6p'],
    'Po': ['5d', '6s', '6p'],
    'At': ['5d', '6s', '6p'],
    'Rn': ['5d', '6s', '6p'],
    'Fr': ['6d', '7s', '7p'],
    'Ra': ['6d', '7s', '7p'],
    'Ac': ['5f', '6d', '7s', '7p'],
    'Th': ['5f', '6d', '7s', '7p'],
    'U' : ['5f', '6d', '7s', '7p'],
    'Np': ['5f', '6d', '7s', '7p'],
    'Pu': ['5f', '6d', '7s', '7p']
}

for key, value in valence_orbitals.items():
    atomic_data[key]['valence_orbitals'] = value


# Set electronic configurations (orbital occupations)
aux=[ ['H', '',{'1s':1}],\
      ['He','',{'1s':2}],\
      # second row
      ['Li','He',{'2s':1,'2p':0}],\
      ['Be','He',{'2s':2,'2p':0}],\
      ['B', 'He',{'2s':2,'2p':1}],\
      ['C', 'He',{'2s':2,'2p':2}],\
      ['N', 'He',{'2s':2,'2p':3}],\
      ['O', 'He',{'2s':2,'2p':4}],\
      ['F', 'He',{'2s':2,'2p':5}],\
      ['Ne','He',{'2s':2,'2p':6}],\
      # third row
      ['Na','Ne',{'3s':1,'3p':0}],\
      ['Mg','Ne',{'3s':2,'3p':0}],\
      ['Al','Ne',{'3s':2,'3p':1}],\
      ['Si','Ne',{'3s':2,'3p':2}],\
      ['P', 'Ne',{'3s':2,'3p':3}],\
      ['S', 'Ne',{'3s':2,'3p':4}],\
      ['Cl','Ne',{'3s':2,'3p':5}],\
      ['Ar','Ne',{'3s':2,'3p':6}],\
      # fourth row
      ['K', 'Ar',{'3d':0,'4s':1,'4p':0}],\
      ['Ca','Ar',{'3d':0,'4s':2,'4p':0}],\
      ['Sc','Ar',{'3d':1,'4s':2,'4p':0}],\
      ['Ti','Ar',{'3d':2,'4s':2,'4p':0}],\
      ['V', 'Ar',{'3d':3,'4s':2,'4p':0}],\
      ['Cr','Ar',{'3d':5,'4s':1,'4p':0}],\
      ['Mn','Ar',{'3d':5,'4s':2,'4p':0}],\
      ['Fe','Ar',{'3d':6,'4s':2,'4p':0}],\
      ['Co','Ar',{'3d':7,'4s':2,'4p':0}],\
      ['Ni','Ar',{'3d':8,'4s':2,'4p':0}],\
      ['Cu','Ar',{'3d':10,'4s':1,'4p':0}],\
      ['Zn','Ar',{'3d':10,'4s':2,'4p':0}],\
      ['Ga','Ar',{'3d':10,'4s':2,'4p':1}],\
      ['Ge','Ar',{'3d':10,'4s':2,'4p':2}],\
      ['As','Ar',{'3d':10,'4s':2,'4p':3}],\
      ['Se','Ar',{'3d':10,'4s':2,'4p':4}],\
      ['Br','Ar',{'3d':10,'4s':2,'4p':5}],\
      ['Kr','Ar',{'3d':10,'4s':2,'4p':6}],\
      # fifth row
      ['Rb','Kr',{'4d':0,'5s':1,'5p':0}],
      ['Sr','Kr',{'4d':0,'5s':2,'5p':0}],
      ['Y', 'Kr',{'4d':1,'5s':2,'5p':0}],
      ['Zr','Kr',{'4d':2,'5s':2,'5p':0}],
      ['Nb','Kr',{'4d':4,'5s':1,'5p':0}],
      ['Mo','Kr',{'4d':5,'5s':1,'5p':0}],
      ['Tc','Kr',{'4d':5,'5s':2,'5p':0}],
      ['Ru','Kr',{'4d':7,'5s':1,'5p':0}],
      ['Rh','Kr',{'4d':8,'5s':1,'5p':0}],
      ['Pd','Kr',{'4d':10,'5s':0,'5p':0}],
      ['Ag','Kr',{'4d':10,'5s':1,'5p':0}],
      ['Cd','Kr',{'4d':10,'5s':2,'5p':0}],
      ['In','Kr',{'4d':10,'5s':2,'5p':1}],
      ['Sn','Kr',{'4d':10,'5s':2,'5p':2}],
      ['Sb','Kr',{'4d':10,'5s':2,'5p':3}],
      ['Te','Kr',{'4d':10,'5s':2,'5p':4}],
      ['I', 'Kr',{'4d':10,'5s':2,'5p':5}],
      ['Xe','Kr',{'4d':10,'5s':2,'5p':6}],
      # sixth row
      ['Cs','Xe',{'5d':0,'6s':1,'6p':0}],
      ['Ba','Xe',{'5d':0,'6s':2,'6p':0}],
      ['Lu','Xe',{'4f':14,'5d':1,'6s':2,'6p':0}],
      ['Hf','Xe',{'4f':14,'5d':2,'6s':2,'6p':0}],
      ['Ta','Xe',{'4f':14,'5d':3,'6s':2,'6p':0}],
      ['W', 'Xe',{'4f':14,'5d':4,'6s':2,'6p':0}],
      ['Re','Xe',{'4f':14,'5d':5,'6s':2,'6p':0}],
      ['Os','Xe',{'4f':14,'5d':6,'6s':2,'6p':0}],
      ['Ir','Xe',{'4f':14,'5d':7,'6s':2,'6p':0}],
      ['Pt','Xe',{'4f':14,'5d':9,'6s':1,'6p':0}],
      ['Au','Xe',{'4f':14,'5d':10,'6s':1,'6p':0}],
      ['Hg','Xe',{'4f':14,'5d':10,'6s':2,'6p':0}], 
      ['Tl','Xe',{'4f':14,'5d':10,'6s':2,'6p':1}], 
      ['Pb','Xe',{'4f':14,'5d':10,'6s':2,'6p':2}], 
      ['Bi','Xe',{'4f':14,'5d':10,'6s':2,'6p':3}], 
      ['Po','Xe',{'4f':14,'5d':10,'6s':2,'6p':4}], 
      ['At','Xe',{'4f':14,'5d':10,'6s':2,'6p':5}], 
      ['Rn','Xe',{'4f':14,'5d':10,'6s':2,'6p':6}], 
      # seventh row
      ['Fr','Rn',{'6d':0,'7s':1,'7p':0}],
      ['Ra','Rn',{'6d':0,'7s':2,'7p':0}],
      ['Ac','Rn',{'5f':0,'6d':1,'7s':2,'7p':0}],
      ['Th','Rn',{'5f':0,'6d':2,'7s':2,'7p':0}],
      ['U', 'Rn',{'5f':3,'6d':1,'7s':2,'7p':0}],
      ['Np','Rn',{'5f':4,'6d':1,'7s':2,'7p':0}],
      ['Pu','Rn',{'5f':6,'6d':0,'7s':2,'7p':0}] ]

configurations = {}
core_configs = {}
for el, core, occu in aux:
    str_conf = ' '.join([f'{orb}{n_el}' for orb, n_el in occu.items()])
    if core != '': 
        configurations[el] = configurations[core].copy()
        core_configs[el] = f'[{core}] {str_conf}'
    else:
        configurations[el] = {}
        core_configs[el] = str_conf
    configurations[el].update(occu)
for key, config in configurations.items():
    atomic_data[key]['configuration'] = config
    atomic_data[key]['core_config'] = core_configs[key]
    atomic_data[key]['valence_number'] = sum([config[orbital] for orbital in atomic_data[key]['valence_orbitals']])

if __name__=='__main__':
    for symbol in atomic_data:
        print('X'*40)
        for d in atomic_data[symbol]:
            print(d, atomic_data[symbol][d])

# SVWN5-LSDA/UGBS reference data for free atomic volumes in a_0^3
# taken from: Kannemann, F. O.; Becke, A. D. J. Chem. Phys. 136, 034109 (2012)
free_volumes = {
    'H': 9.194,
    'He': 4.481,
    'Li': 91.96,
    'Be': 61.36,
    'B': 49.81,
    'C': 36.73,
    'N': 27.63,
    'O': 23.52,
    'F': 19.32,
    'Ne': 15.95,
    'Na': 109.4,
    'Mg': 103.1,
    'Al': 120.4,
    'Si': 104.2,
    'P': 86.78,
    'S': 77.13,
    'Cl': 66.37,
    'Ar': 57.34,
    'K': 203.1,
    'Ca': 212.2,
    'Sc': 183.1,
    'Ti': 162.3,
    'V': 143.2,
    'Cr': 108.2,
    'Mn': 123.1,
    'Fe': 105.7,
    'Co': 92.94,
    'Ni': 83.79,
    'Cu': 75.75,
    'Zn': 81.18,
    'Ga': 118.4,
    'Ge': 116.3,
    'As': 107.5,
    'Se': 103.2,
    'Br': 95.11,
    'Kr': 87.61,
    'Rb': 248.8,
    'Sr': 273.7,
    'Y': 249.2,
    'Zr': 223.8,
    'Nb': 175.8,
    'Mo': 156.8,
    'Tc': 160.0,
    'Ru': 136.7,
    'Rh': 127.8,
    'Pd': 97.02,
    'Ag': 112.8,
    'Cd': 121.6,
    'In': 167.9,
    'Sn': 172.0,
    'Sb': 165.5,
    'Te': 163.0,
    'I': 154.0,
    'Xe': 146.1,
    'Cs': 342.0,
    'Ba': 385.8,
    'La': 343.4,
    'Ce': 350.3,
    'Pr': 334.9,
    'Nd': 322.2,
    'Pm': 310.3,
    'Sm': 299.5,
    'Eu': 289.6,
    'Gd': 216.1,
    'Tb': 268.9,
    'Dy': 259.8,
    'Ho': 251.3,
    'Er': 243.2,
    'Tm': 235.5,
    'Yb': 228.3,
    'Lu': 229.6,
    'Hf': 210.0,
    'Ta': 197.5,
    'W': 183.2,
    'Re': 174.7,
    'Os': 164.1,
    'Ir': 150.4,
    'Pt': 135.8,
    'Au': 125.3,
    'Hg': 131.3,
    'Tl': 185.8,
    'Pb': 195.7,
    'Bi': 193.0,
    'Po': 189.1,
    'At': 185.9,
    'Rn': 181.1,
    'Fr': 357.8,
    'Ra': 407.3,
    'Ac': 383.1,
    'Th': 362.1,
    'Pa': 346.6,
    'U': 332.5,
    'Np': 319.6,
    'Pu': 308.1,
    'Am': 297.4,
    'Cm': 300.6,
    'Bk': 275.8,
    'Cf': 266.3,
    'Es': 257.4,
    'Fm': 209.7,
    'Md': 203.2,
    'No': 230.2,
    'Lr': 236.9
}

## Occupation of valence orbitals in free atoms (non-zero occupations only)
ValOccs_lm_free = {
    'H': {'s': 1.},
    'He': {'s': 2.},
    'Li': {'s': 1., 'p': 0.},
    'Be': {'s': 2., 'p': 0.},
    'B': {'s': 2., 'p': 0.33333333},
    'C': {'s': 2., 'p': 0.66666666},
    'N': {'s': 2., 'p': 1.},
    'O': {'s': 2., 'p': 1.33333333},
    'F': {'s': 2., 'p': 1.66666666},
    'Ne': {'s': 2., 'p': 2.},
    'Na': {'s': 1., 'p': 0.},
    'Mg': {'s': 2., 'p': 0.},
    'Al': {'s': 2., 'p': 0.33333333},
    'Si': {'s': 2., 'p': 0.66666666},
    'P': {'s': 2., 'p': 1.},
    'S': {'s': 2., 'p': 1.33333333},
    'Cl': {'s': 2., 'p': 1.66666666},
    'Ar': {'s': 2., 'p': 2.},
    'K': {'s': 1., 'p': 0.},
    'Ca': {'s': 2., 'p': 0.},
    'Sc': {'s': 2., 'd': 0.2},
    'Ti': {'s': 2., 'd': 0.4},
    'V': {'s': 2., 'd': 0.6},
    'Cr': {'s': 1., 'd': 1.},
    'Mn': {'s': 2., 'd': 1.},
    'Fe': {'s': 2., 'd': 1.2},
    'Co': {'s': 2., 'd': 1.4},
    'Ni': {'s': 2., 'd': 1.6},
    'Cu': {'s': 1., 'd': 2.},
    'Zn': {'s': 2., 'd': 2.},
    'Ga': {'s': 2., 'p': 0.33333333},
    'Ge': {'s': 2., 'p': 0.66666666},
    'As': {'s': 2., 'p': 1.66666666},
    'Se': {'s': 2., 'p': 1.33333333},
    'Br': {'s': 2., 'p': 1.66666666},
    'Kr': {'s': 2., 'p': 2.},
    'Rb': {'s': 1., 'p': 0.},
    'Sr': {'s': 2., 'p': 0.},
    'Y': {'s': 2., 'p': 0., 'd': 0.2},
    'Zr': {'s': 2., 'p': 0., 'd': 0.4},
    'Nb': {'s': 1., 'p': 0., 'd': 0.8},
    'Mo': {'s': 1., 'p': 0., 'd': 1.},
    'Tc': {'s': 1., 'p': 0., 'd': 1.2},
    'Ru': {'s': 1., 'p': 0., 'd': 1.4},
    'Rh': {'s': 1., 'p': 0., 'd': 1.6},
    'Pd': {'s': 0., 'p': 0., 'd': 2.},
    'Ag': {'s': 1., 'p': 0., 'd': 2.},
    'Cd': {'s': 2., 'p': 0., 'd': 2.},
    'In': {'s': 2., 'p': 0.33333333},
    'Sn': {'s': 2., 'p': 0.66666666},
    'Sb': {'s': 2., 'p': 1.},
    'Te': {'s': 2., 'p': 1.33333333},
    'I': {'s': 2., 'p': 1.66666666},
    'Xe': {'s': 2., 'p': 2.},
    'Cs': {'s': 1., 'p': 0.},
    'Ba': {'s': 2., 'p': 0.},
    'Lu': {'s': 2., 'p': 0., 'd': 0.2},
    'Hf': {'s': 2., 'p': 0., 'd': 0.4},
    'Ta': {'s': 2., 'p': 0., 'd': 0.6},
    'W': {'s': 2., 'p': 0., 'd': 0.8},
    'Re': {'s': 2., 'p': 0., 'd': 1.},
    'Os': {'s': 2., 'p': 0., 'd': 1.2},
    'Ir': {'s': 2., 'p': 0., 'd': 1.4},
    'Pt': {'s': 1., 'p': 0., 'd': 1.8},
    'Au': {'s': 1., 'p': 0., 'd': 2.},
    'Hg': {'s': 2., 'p': 0., 'd': 2.},
    'Tl': {'s': 2., 'p': 0.33333333},
    'Pb': {'s': 2., 'p': 0.66666666},
    'Bi': {'s': 2., 'p': 1.},
    'Po': {'s': 2., 'p': 1.33333333},
    'As': {'s': 2., 'p': 1.66666666},
    'Rn': {'s': 2., 'p': 2.}
}

## optimal confinement parameters (optimized for band structures of homonuclear crystals)
## from M. Wahiduzzaman, et al. J. Chem. Theory Comput. 9, 4006-4017 (2013)
## general form V_conf = (r/r0)**s, r0 in Bohr
conf_parameters = {
    'H': {'r0': 1.6, 's': 2.2},
    'He': {'r0': 1.4, 's': 11.4},
    'Li': {'r0': 5.0, 's': 8.2},
    'Be': {'r0': 3.4, 's': 13.2},
    'B': {'r0': 3.0, 's': 10.4},
    'C': {'r0': 3.2, 's': 8.2},
    'N': {'r0': 3.4, 's': 13.4},
    'O': {'r0': 3.1, 's': 12.4},
    'F': {'r0': 2.7, 's': 10.6},
    'Ne': {'r0': 3.2, 's': 15.4},
    'Na': {'r0': 5.9, 's': 12.6},
    'Mg': {'r0': 5.0, 's': 6.2},
    'Al': {'r0': 5.9, 's': 12.4},
    'Si': {'r0': 4.4, 's': 12.8},
    'P': {'r0': 4.0, 's': 9.6},
    'S': {'r0': 3.9, 's': 4.6},
    'Cl': {'r0': 3.8, 's': 9.0},
    'Ar': {'r0': 4.5, 's': 15.2},
    'K': {'r0': 6.5, 's': 15.8},
    'Ca': {'r0': 4.9, 's': 13.6},
    'Sc': {'r0': 5.1, 's': 13.6},
    'Ti': {'r0': 4.2, 's': 12.0},
    'V': {'r0': 4.3, 's': 13.0},
    'Cr': {'r0': 4.7, 's': 3.6},
    'Mn': {'r0': 3.6, 's': 11.6},
    'Fe': {'r0': 3.7, 's': 11.2},
    'Co': {'r0': 3.3, 's': 11.0},
    'Ni': {'r0': 3.7, 's': 2.2},
    'Cu': {'r0': 5.2, 's': 2.2},
    'Zn': {'r0': 4.6, 's': 2.2},
    'Ga': {'r0': 5.9, 's': 8.8},
    'Ge': {'r0': 4.5, 's': 13.4},
    'As': {'r0': 4.4, 's': 5.6},
    'Se': {'r0': 4.5, 's': 3.8},
    'Br': {'r0': 4.3, 's': 6.4},
    'Kr': {'r0': 4.8, 's': 15.6},
    'Rb': {'r0': 9.1, 's': 16.8},
    'Sr': {'r0': 6.9, 's': 14.8},
    'Y': {'r0': 5.7, 's': 13.6},
    'Zr': {'r0': 5.2, 's': 14.0},
    'Nb': {'r0': 5.2, 's': 15.0},
    'Mo': {'r0': 4.3, 's': 11.6},
    'Tc': {'r0': 4.1, 's': 12.0},
    'Ru': {'r0': 4.1, 's': 3.8},
    'Rh': {'r0': 4.0, 's': 3.4},
    'Pd': {'r0': 4.4, 's': 2.8},
    'Ag': {'r0': 6.5, 's': 2.0},
    'Cd': {'r0': 5.4, 's': 2.0},
    'In': {'r0': 4.8, 's': 13.2},
    'Sn': {'r0': 4.7, 's': 13.4},
    'Sb': {'r0': 5.2, 's': 3.0},
    'Te': {'r0': 5.2, 's': 3.0},
    'I': {'r0': 6.2, 's': 2.0},
    'Xe': {'r0': 5.2, 's': 16.2},
    'Cs': {'r0': 10.6, 's': 13.6},
    'Ba': {'r0': 7.7, 's': 12.0},
    'La': {'r0': 7.4, 's': 8.6},
    'Lu': {'r0': 5.9, 's': 16.4},
    'Hf': {'r0': 5.2, 's': 14.8},
    'Ta': {'r0': 4.8, 's': 13.8},
    'W': {'r0': 4.2, 's': 8.6},
    'Re': {'r0': 4.2, 's': 13.0},
    'Os': {'r0': 4.0, 's': 8.0},
    'Ir': {'r0': 3.9, 's': 12.6},
    'Pt': {'r0': 3.8, 's': 12.8},
    'Au': {'r0': 4.8, 's': 2.0},
    'Hg': {'r0': 6.7, 's': 2.0},
    'Tl': {'r0': 7.3, 's': 2.2},
    'Pb': {'r0': 5.7, 's': 3.0},
    'Bi': {'r0': 5.8, 's': 2.6},
    'Po': {'r0': 5.5, 's': 2.2},
    'Ra': {'r0': 7.0, 's': 14.0},
    'Th': {'r0': 6.2, 's': 4.4}
}

for key, conf_param in conf_parameters.items():
    conf_param['mode'] = 'general'


## shell resolved U-parameters (as obtained from PBE-DFT calculations)
## from M. Wahiduzzaman, et al. J. Chem. Theory Comput. 9, 4006-4017 (2013)
## using U parameter of occupied shell with highest l for unoccupied shells
U_parameters = {
    'H' : {'d': 0.419731, 'p': 0.419731, 's': 0.419731},
    'He': {'d': 0.742961, 'p': 0.742961, 's': 0.742961},
    'Li': {'d': 0.131681, 'p': 0.131681, 's': 0.174131},
    'Be': {'d': 0.224651, 'p': 0.224651, 's': 0.270796},
    'B' : {'d': 0.296157, 'p': 0.296157, 's': 0.333879},
    'C' : {'d': 0.364696, 'p': 0.364696, 's': 0.399218},
    'N' : {'d': 0.430903, 'p': 0.430903, 's': 0.464356},
    'O' : {'d': 0.495405, 'p': 0.495405, 's': 0.528922},
    'F' : {'d': 0.558631, 'p': 0.558631, 's': 0.592918},
    'Ne': {'d': 0.620878, 'p': 0.620878, 's': 0.656414},
    'Na': {'d': 0.087777, 'p': 0.087777, 's': 0.165505},
    'Mg': {'d': 0.150727, 'p': 0.150727, 's': 0.224983},
    'Al': {'d': 0.186573, 'p': 0.203216, 's': 0.261285},
    'Si': {'d': 0.196667, 'p': 0.247841, 's': 0.300005},
    'P' : {'d': 0.206304, 'p': 0.289262, 's': 0.338175},
    'S' : {'d': 0.212922, 'p': 0.328724, 's': 0.37561},
    'Cl': {'d': 0.214242, 'p': 0.366885, 's': 0.412418},
    'Ar': {'d': 0.207908, 'p': 0.404106, 's': 0.448703},
    'K' : {'d': 0.171297, 'p': 0.081938, 's': 0.136368},
    'Ca': {'d': 0.299447, 'p': 0.128252, 's': 0.177196},
    'Sc': {'d': 0.32261,  'p': 0.137969, 's': 0.189558},
    'Ti': {'d': 0.351019, 'p': 0.144515, 's': 0.201341},
    'V' : {'d': 0.376535, 'p': 0.149029, 's': 0.211913},
    'Cr': {'d': 0.31219,  'p': 0.123012, 's': 0.200284},
    'Mn': {'d': 0.422038, 'p': 0.155087, 's': 0.23074},
    'Fe': {'d': 0.442914, 'p': 0.156593, 's': 0.239398},
    'Co': {'d': 0.462884, 'p': 0.157219, 's': 0.24771},
    'Ni': {'d': 0.401436, 'p': 0.10618,  's': 0.235429},
    'Cu': {'d': 0.42067,  'p': 0.097312, 's': 0.243169},
    'Zn': {'d': 0.518772, 'p': 0.153852, 's': 0.271212},
    'Ga': {'d': 0.051561, 'p': 0.205025, 's': 0.279898},
    'Ge': {'d': 0.101337, 'p': 0.240251, 's': 0.304342},
    'As': {'d': 0.127856, 'p': 0.271613, 's': 0.330013},
    'Se': {'d': 0.165858, 'p': 0.300507, 's': 0.355433},
    'Br': {'d': 0.189059, 'p': 0.327745, 's': 0.380376},
    'Kr': {'d': 0.200972, 'p': 0.353804, 's': 0.404852},
    'Rb': {'d': 0.180808, 'p': 0.07366,  's': 0.130512},
    'Sr': {'d': 0.234583, 'p': 0.115222, 's': 0.164724},
    'Y' : {'d': 0.239393, 'p': 0.127903, 's': 0.176814},
    'Zr': {'d': 0.269067, 'p': 0.136205, 's': 0.189428},
    'Nb': {'d': 0.294607, 'p': 0.141661, 's': 0.20028},
    'Mo': {'d': 0.317562, 'p': 0.145599, 's': 0.209759},
    'Tc': {'d': 0.338742, 'p': 0.148561, 's': 0.218221},
    'Ru': {'d': 0.329726, 'p': 0.117901, 's': 0.212289},
    'Rh': {'d': 0.350167, 'p': 0.113146, 's': 0.219321},
    'Pd': {'d': 0.369605, 'p': 0.107666, 's': 0.225725},
    'Ag': {'d': 0.388238, 'p': 0.099994, 's': 0.231628},
    'Cd': {'d': 0.430023, 'p': 0.150506, 's': 0.251776},
    'In': {'d': 0.156519, 'p': 0.189913, 's': 0.257192},
    'Sn': {'d': 0.171708, 'p': 0.217398, 's': 0.275163},
    'Sb': {'d': 0.184848, 'p': 0.241589, 's': 0.294185},
    'Te': {'d': 0.195946, 'p': 0.263623, 's': 0.313028},
    'I' : {'d': 0.206534, 'p': 0.284168, 's': 0.33146},
    'Xe': {'d': 0.211949, 'p': 0.303641, 's': 0.349484},
    'Cs': {'d': 0.159261, 'p': 0.06945,  's': 0.12059},
    'Ba': {'d': 0.199559, 'p': 0.105176, 's': 0.149382},
    'La': {'d': 0.220941, 'p': 0.115479, 's': 0.160718},
    'Lu': {'d': 0.220882, 'p': 0.12648,  's': 0.187365},
    'Hf': {'d': 0.249246, 'p': 0.135605, 's': 0.200526},
    'Ta': {'d': 0.273105, 'p': 0.141193, 's': 0.212539},
    'W':  {'d': 0.294154, 'p': 0.144425, 's': 0.223288},
    'Re': {'d': 0.313288, 'p': 0.146247, 's': 0.233028},
    'Os': {'d': 0.331031, 'p': 0.146335, 's': 0.241981},
    'Ir': {'d': 0.347715, 'p': 0.145121, 's': 0.250317},
    'Pt': {'d': 0.363569, 'p': 0.143184, 's': 0.258165},
    'Au': {'d': 0.361156, 'p': 0.090767, 's': 0.255962},
    'Hg': {'d': 0.393392, 'p': 0.134398, 's': 0.272767},
    'Tl': {'d': 0.11952,  'p': 0.185496, 's': 0.267448},
    'Pb': {'d': 0.128603, 'p': 0.209811, 's': 0.280804},
    'Bi': {'d': 0.14221,  'p': 0.231243, 's': 0.296301},
    'Po': {'d': 0.158136, 'p': 0.250546, 's': 0.311976},
    'Ra': {'d': 0.167752, 'p': 0.093584, 's': 0.151368},
    'Th': {'d': 0.21198,  'p': 0.114896, 's': 0.174221}
}

for key, value in U_parameters.items():
    if key in atomic_data.keys():
        atomic_data[key]['U'] = value


## Orbital energies for the neutral atoms
## from M. Wahiduzzaman, et al. J. Chem. Theory Comput. 9, 4006-4017 (2013)
E_values = {
    'H':  {'s': -0.238603},
    'He': {'s': -0.579318},
    'Li': {'p': -0.040054, 's': -0.105624},
    'Be': {'p': -0.074172, 's': -0.206152},
    'B':  {'p': -0.132547, 's': -0.347026},
    'C':  {'p': -0.194236, 's': -0.505337},
    'N':  {'p': -0.260544, 's': -0.682915},
    'O':  {'p': -0.331865, 's': -0.880592},
    'F':  {'p': -0.408337, 's': -1.098828},
    'Ne': {'p': -0.490009, 's': -1.337930},
    'Na': {'p': -0.027320, 's': -0.100836},
    'Mg': {'p': -0.048877, 's': -0.172918},
    'Al': {'d': 0.116761, 'p': -0.099666, 's': -0.284903},
    'Si': {'d': 0.113134, 'p': -0.149976, 's': -0.397349},
    'P':  {'d': 0.121111, 'p': -0.202363, 's': -0.513346},
    'S':  {'d': 0.134677, 'p': -0.257553, 's': -0.634144},
    'Cl': {'d': 0.150683, 'p': -0.315848, 's': -0.760399},
    'Ar': {'d': 0.167583, 'p': -0.377389, 's': -0.892514},
    'K':  {'d': 0.030121, 'p': -0.029573, 's': -0.085219},
    'Ca': {'d': -0.070887, 'p': -0.051543, 's': -0.138404},
    'Sc': {'d': -0.118911, 'p': -0.053913, 's': -0.153708},
    'Ti': {'d': -0.156603, 'p': -0.053877, 's': -0.164133},
    'V':  {'d': -0.189894, 'p': -0.053055, 's': -0.172774},
    'Cr': {'d': -0.107113, 'p': -0.036319, 's': -0.147221},
    'Mn': {'d': -0.248949, 'p': -0.050354, 's': -0.187649},
    'Fe': {'d': -0.275927, 'p': -0.048699, 's': -0.194440},
    'Co': {'d': -0.301635, 'p': -0.046909, 's': -0.200975},
    'Ni': {'d': -0.170792, 'p': -0.027659, 's': -0.165046},
    'Cu': {'d': -0.185263, 'p': -0.025621, 's': -0.169347},
    'Zn': {'d': -0.372826, 'p': -0.040997, 's': -0.219658},
    'Ga': {'d': 0.043096, 'p': -0.094773, 's': -0.328789},
    'Ge': {'d': 0.062123, 'p': -0.143136, 's': -0.431044},
    'As': {'d': 0.078654, 'p': -0.190887, 's': -0.532564},
    'Se': {'d': 0.104896, 'p': -0.239256, 's': -0.635202},
    'Br': {'d': 0.126121, 'p': -0.288792, 's': -0.739820},
    'Kr': {'d': 0.140945, 'p': -0.339778, 's': -0.846921},
    'Rb': {'d': 0.030672, 'p': -0.027523, 's': -0.081999},
    'Sr': {'d': -0.041363, 'p': -0.047197, 's': -0.129570},
    'Y':  {'d': -0.092562, 'p': -0.052925, 's': -0.150723},
    'Zr': {'d': -0.132380, 'p': -0.053976, 's': -0.163093},
    'Nb': {'d': -0.170468, 'p': -0.053629, 's': -0.172061},
    'Mo': {'d': -0.207857, 'p': -0.052675, 's': -0.179215},
    'Tc': {'d': -0.244973, 'p': -0.051408, 's': -0.185260},
    'Ru': {'d': -0.191289, 'p': -0.033507, 's': -0.155713},
    'Rh': {'d': -0.218418, 'p': -0.031248, 's': -0.157939},
    'Pd': {'d': -0.245882, 'p': -0.029100, 's': -0.159936},
    'Ag': {'d': -0.273681, 'p': -0.027061, 's': -0.161777},
    'Cd': {'d': -0.431379, 'p': -0.043481, 's': -0.207892},
    'In': {'d': 0.135383, 'p': -0.092539, 's': -0.301650},
    'Sn': {'d': 0.125834, 'p': -0.135732, 's': -0.387547},
    'Sb': {'d': 0.118556, 'p': -0.177383, 's': -0.471377},
    'Te': {'d': 0.114419, 'p': -0.218721, 's': -0.555062},
    'I':  {'d': 0.112860, 'p': -0.260330, 's': -0.639523},
    'Xe': {'d': 0.111715, 'p': -0.302522, 's': -0.725297},
    'Cs': {'d': -0.007997, 'p': -0.027142, 's': -0.076658},
    'Ba': {'d': -0.074037, 'p': -0.045680, 's': -0.118676},
    'La': {'d': -0.113716, 'p': -0.049659, 's': -0.135171},
    'Lu': {'d': -0.064434, 'p': -0.049388, 's': -0.171078},
    'Hf': {'d': -0.098991, 'p': -0.051266, 's': -0.187557},
    'Ta': {'d': -0.132163, 'p': -0.051078, 's': -0.199813},
    'W':  {'d': -0.164874, 'p': -0.049978, 's': -0.209733},
    'Re': {'d': -0.197477, 'p': -0.048416, 's': -0.218183},
    'Os': {'d': -0.230140, 'p': -0.046602, 's': -0.225640},
    'Ir': {'d': -0.262953, 'p': -0.044644, 's': -0.232400},
    'Pt': {'d': -0.295967, 'p': -0.042604, 's': -0.238659},
    'Au': {'d': -0.252966, 'p': -0.028258, 's': -0.211421},
    'Hg': {'d': -0.362705, 'p': -0.038408, 's': -0.250189},
    'Tl': {'d': 0.081292, 'p': -0.087069, 's': -0.350442},
    'Pb': {'d': 0.072602, 'p': -0.128479, 's': -0.442037},
    'Bi': {'d': 0.073863, 'p': -0.167900, 's': -0.531518},
    'Po': {'d': 0.081795, 'p': -0.206503, 's': -0.620946},
    'Ra': {'d': -0.047857, 'p': -0.037077, 's': -0.120543},
    'Th': {'d': -0.113604, 'p': -0.045825, 's': -0.161992}
}

for key, value in E_values.items():
    if key in atomic_data:
        atomic_data[key]['E'] = value


## reference values for C6_AA for free atoms in Ha*Bohr**6
C6_ref = { 'H':6.50,   'He':1.46, \
          'Li':1387.0, 'Be':214.0,   'B':99.5,    'C':46.6,    'N':24.2,   'O':15.6,   'F':9.52,   'Ne':6.38, \
          'Na':1556.0, 'Mg':627.0,  'Al':528.0,  'Si':305.0,   'P':185.0 , 'S':134.0, 'Cl':94.6,   'Ar':64.3, \
           'K':3897.0, 'Ca':2221.0, 'Sc':1383.0, 'Ti':1044.0,  'V':832.0, 'Cr':602.0, 'Mn':552.0,  'Fe':482.0, \
          'Co':408.0,  'Ni':373.0,  'Cu':253.0,  'Zn':284.0,  'Ga':498.0, 'Ge':354.0, 'As':246.0,  'Se':210.0, \
          'Br':162.0,  'Kr':129.6,  'Rb':4691.0, 'Sr':3170.0, 'Rh':469.0, 'Pd':157.5, 'Ag':339.0,  'Cd':452.0, \
          'In':779.0,  'Sn':659.0,  'Sb':492.0,  'Te':396.0,   'I':385.0, 'Xe':285.9, 'Ba':5727.0, 'Ir':359.1, \
          'Pt':347.1,  'Au':298.0,  'Hg':392.0,  'Pb':697.0,  'Bi':571.0 }

## reference values for static polarizabilities in Bohr**3
alpha0_ref = { 'H':4.50,  'He':1.38, \
              'Li':164.2, 'Be':38.0,   'B':21.0,    'C':12.0,    'N':7.4,   'O':5.4,   'F':3.8,   'Ne':2.67, \
              'Na':162.7, 'Mg':71.0,  'Al':60.0,  'Si':37.0,   'P':25.0 , 'S':19.6, 'Cl':15.0,   'Ar':11.1, \
               'K':292.9, 'Ca':160.0, 'Sc':120.0, 'Ti':98.0,  'V':84.0, 'Cr':78.0, 'Mn':63.0,  'Fe':56.0, \
              'Co':50.0,  'Ni':48.0,  'Cu':42.0,  'Zn':40.0,  'Ga':60.0, 'Ge':41.0, 'As':29.0,  'Se':25.0, \
              'Br':20.0,  'Kr':16.8,  'Rb':319.2, 'Sr':199.0, 'Rh':56.1, 'Pd':23.68, 'Ag':50.6,  'Cd':39.7, \
              'In':75.0,  'Sn':60.0,  'Sb':44.0,  'Te':37.65,   'I':35.0, 'Xe':27.3, 'Ba':275.0, 'Ir':42.51, \
              'Pt':39.68,  'Au':36.5,  'Hg':33.9,  'Pb':61.8,  'Bi':49.02 }
              
## this is currently only dummy dictionary
R_conf = {}

## additional bulk reference data (from http://www.webelements.com)
## lattice constants in Angstroms, extends ase.data.reference_states
additional_bulk = {}
additional_bulk['C']  = {'hcp':{'a':2.464, 'c':6.711}}
additional_bulk['Ge'] = {'fcc':{'a':5.6575}}
additional_bulk['Sn'] = {'diamond':{'a':6.48920}}

#--EOF--#
