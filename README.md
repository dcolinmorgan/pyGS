# JATNIpy #

fork of Justin's, Andrea's, and Torbjörn's network inference package for Python

[Nordlinglab](https://www.nordlinglab.org/)
[JATNIpy](https://bitbucket.org/temn/JATNIpy/src/)

## What is JATNIpy? ##
We re-implemented the [GeneSPIDER](https://bitbucket.org/sonnhammergrni/genespider/src/) toolbox and chose Python as our programming language. Python is a popular high-level programming language. It is freely available and widely used by academic and commercial.



### Method: ### 
We incorporate several free available python packages into a new complete package. We finally name the new complete package as JATNIpy. 

Availability and Implementation: The source code is freely available for download at https://bitbucket.org/temn/JATNIpy/. 
It's a reimplementation of the GeneSPIDER toolbox in Python.

Contact: [Justin](mailto:justin.lin@nordlinglab.org)


### How to set it up? ###

Before we uses JATNIpy, we should make sure pip3 has been installed in local computer. To install pip, use the following command:
* For Debian/Ubuntu user:
```
apt-get install python3-pip
```

* For CentOS 7 user:
```
yum install python34-setuptools
easy_install pip
```

Having pip3 in our local computer, we then install virtualenvwrapper to create a virtual environment for our local computer by these commands
```	
pip3 install virtualenvwrapper
export WORKON_HOME=~/Envs
mkdir -p $WORKON_HOME
source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv env1
workon env2
```
Create the new virtual environment `env1` by 
```
mkvirtualenv env1
```
Choose the virtual environment you want to work on by 
```
workon env1
```

#### For downloading and installing `JATNIpy`

##### Alternative 1 : GIT
1. Use [git](https://git-scm.com/) to fetch JATNIpy repository run this command
```
git clone https://bitbucket.org/temn/JATNIpy/
```
2. Change to the directory where you downloaded from the repository by
```
cd ~/JATNIpy/jatnipy.
```
3. After working on the environment you want, then use pip3 to install
the open source python3 packages with the command
```
pip3 install -e
``` 
##### Alternative 2 : Direct download
1. Download it from [JATNIpy](https://bitbucket.org/temn/JATNIpy/)
2. Change to the directory where you downloaded from the repository by
```
cd ~/JATNIpy/jatnipy.
```
3. After working on the environment you want, then use pip3 to install
the open source python3 packages with the command
```
pip3 install -e
``` 

##### Alternative 3 : Direct installation
Use the pip3 to install JATNIpy from [Pypi](https://pypi.org/)
```
pip3 install jatnipy -t ~/JATNIpy
```

In alternatives 1 and 2, when using `pip3 install -e`, it installs the following dependencies:

* [git](https://git-scm.com/) Version control system for tracking the development of programming
* [Scipy](https://www.scipy.org/) Python-based software for mathematics, science, and engineering	
* [Numpy](http://www.numpy.org/) Fundamental python package for doing numerical or mathematics computation
* [pandas](https://pandas.pydata.org/) Using data structures and data analysis tools easily in Python	
* [matplotlib](https://matplotlib.org/) Useful Python 2D plotting tool which provides MATLAB-like interface	
* [scikit-learn](http://scikit-learn.org/stable/) Data mining and data analysis which built on Numpy, Scipy and matplotlib	
* [networkx](https://networkx.github.io/) Python package which is made for studying graphs and the complex networks	
* [glmnet_py](https://pypi.org/project/glmnet-py/) The popular glmnet library for Python version
* [py-ubjson](https://pypi.org/project/py-ubjson/) Universal Binary JSON encoder/decoder for Python version	
* [CVXPY](http://www.cvxpy.org/) Handling convex optimization problems for Python version	
* [Requests: HTTP for Humans](http://docs.python-requests.org/en/master/) Python library for requesting HTTP

#### Dataset and Network

* Datasets are available [here](https://bitbucket.org/sonnhammergrni/gs-datasets).
* Networks are available [here](https://bitbucket.org/sonnhammergrni/gs-networks).

#### Basic functionality

To test that basic functionality after setup, open python3 and run the commands:
Network fetching:
```python=
# for better appearance
from pprint import pprint as p 
from datastruct.Network import Network as N
# To show the local folder content 
N.Load('.')
# To initialize the network
N.Load('test') 
# To fetch the network from the website
N.Load('random/N10/','https://bitbucket.org/api/2.0/repositories/sonnhammergrni/gs-networks/src/master') 
```
Dataset fetching:
```python=10
from datastruct.Dataset import Dataset as D
# To show the local folder content
D.Load('.') 
# To initialize the dataset 
D.Load('test') 
# To fetch the dataset from the website
D.Load('N10','https://bitbucket.org/api/2.0/repositories/sonnhammergrni/gs-datasets/src/master/') 
```
The structures from above, network and dataset, needs to be converted to Python objects for :
```python=17
from datastruct.Dataset import Dataset
Da = Dataset()
Data = Dataset(Dataset.Load('test'))
from datastruct.Network import Network
Ne = Network()
Net = Network(Network.Load('test'))
```
You should now have the default network and a dataset loaded in python.

Generating example data as used in results section
--------------------------------------------------
Same as with the GeneSPIDER toolbox, the network and dataset which we used in the examples can be downloaded from the online repository at https://bitbucket.org/sonnhammergrni/gs-networks and https://bitbucket.org/sonnhammergrni/gs-datasets. 
In addition, the example network and dataset we used are [Nordling-D20100302-random-N10-L25-ID1446937](https://bitbucket.org/sonnhammergrni/gs-networks/raw/ec384db2750b5ef229d1c613e3dd04a5e3b634e2/random/N10/Nordling-D20100302-random-N10-L25-ID1446937.json) and [Nordling-ID1446937-D20150825-E15-SNR3291-IDY15968](https://bitbucket.org/sonnhammergrni/gs-datasets/raw/a9d9b00aaa5fa6f4059ba03fd0cb5ec8eb80f3f0/N10/Nordling-ID1446937-D20150825-E15-SNR3291-IDY15968.json), respectively. The code below is to generate a new network and dataset which will differ from the examples. Because we use the random number generators to create the network and noise matrices.

### Network generation in JATNIpy

Next, we generate a stable random network with 10 nodes and [sparsity](https://en.wikipedia.org/wiki/Dense_graph#Sparse_and_tight_graphs) 0.25. With these specifications, the following code creates a `datastruct.Network` object.

```python=
from datastruct.randomNet import randomNet
from datastruct.stabilize import stabilize
from datastruct.Network import Network	
Ne = Network()
import numpy as np
# Defining number of nodes
N = 10
# Defining the sparsity
S = 0.25
A = randomNet(N,S)-np.eye(N)
A = stabilize(A,'iaa','high')
Net = Ne.Network(A,'random')
creator_setting = {'creator':'Nordlinglab_Justin'}
Ne.setname(Net,creator_setting)
Net.description = 'This is a sparse network with 10 nodes, 10 negative self-loops and 15 randomly chosen links generated by Justin 2018-08-31. The coefficients are chosen such that they form one strong component and a stable dynamical system with time constants in the range 0.089 to 12 and an interampatteness level of 145 that is in-between the estimated level of an E. coli (Gardner et al. 2003 Science) and Yeast (Lorenz et al. 2009 PNAS) gene regulatory network. The coefficients of the network have not been tuned to explain any of the data sets in the mentioned articles.'
```

The random network created using `datastruct.randomNet` and the desired interampatteness (IAA) degree are used as input parameters to put in the `datastruct.stabilize`. This one stabilizes the network by making the real part of all eigenvalues negative while adjusting the IAA degree level. The method `Ne.setname` is used to specify the fields of the Network object. The network's name is generated based on its properties automatically to make sure that each one is unique. Then, you can run the command to save your Network in the `.json` format using `savepath` (the directory where you want to save).
```python=16
Network.save(Network,Net,'savepath/','.json')
``` 
After saving the Network object, you can use the next commands to display the content of the `Network` object that you have generated.

```python=17
Net = Network(Network.Load('./savepath/Nordlinglab_Justin-D20190414-random-N10-L12ID33097.json'))
Net.__dict__
```

The content displayed is shown below. It shows the properties of the `Network` object. Some of the content are:
* `A`: is the network matrix
* `G`: is the static gain matrix
* `N`: is the number of nodes
* `created`: is the dictionary record the detail information for this object
* `description`: is a description of the network
* `name`: is the name of the Network object which contains the name of the creator `Nordlinglab_Justin`, the date of creation `D` and the number of edges `L`
* `names`,`nodes`: contain the name assigned to each node, which are generated automatically if they are not specified
* `params` records nonzeros in the network matrix
* `random`: is the type of network
* `shape` is the shape of the network matrix
* `structure` which is same as type is `random`  
```
{'A': source        G01       G02       G03    ...           G08       G09       G10
target                                   ...                                  
G01     -9.999993  0.000000  0.000000    ...    -34.951935  0.000000  0.000000
G02      0.000000 -9.999993  0.000000    ...      0.000000  0.000000  0.000000
G03      0.000000  0.000000 -9.999993    ...      0.000000  0.000000  0.000000
G04      0.000000  0.000000  0.000000    ...      0.000000  0.000000  0.000000
G05      0.000000  0.000000  0.000000    ...      0.000000  0.000000  0.000000
G06    -46.352859  0.000000  0.000000    ...      0.000000  0.000000  0.000000
G07      0.000000  0.000000  0.000000    ...      0.000000  0.000000  0.000000
G08     34.951935  0.000000  0.000000    ...     -9.999993  0.000000  0.000000
G09      0.000000  0.000000  0.000000    ...      0.000000 -9.999993  0.000000
G10      0.000000  0.000000  0.000000    ...      0.000000  0.000000 -9.999993

[10 rows x 10 columns],
'G': source           G01           G02           G03 ...            G08  G09  G10
target                                           ...                         
G01     2.881656e-03  7.357080e-20 -0.000000e+00 ...  -1.007195e-02 -0.0 -0.0
G02    -5.187064e-18  1.000001e-01 -1.230312e-17 ...   2.462769e-17 -0.0 -0.0
G03     1.054544e-17  8.764060e-18  1.000001e-01 ...   1.465668e-17 -0.0 -0.0
G04    -1.924527e-18  7.464553e-18 -2.235816e-18 ...  -1.955644e-19 -0.0 -0.0
G05    -1.574097e-33  9.320219e-18 -2.277998e-18 ...   1.414714e-33 -0.0 -0.0
G06    -1.335731e-02 -1.022088e-17  1.552109e-17 ...   4.668640e-02 -0.0 -0.0
G07    -0.000000e+00 -0.000000e+00 -0.000000e+00 ...  -0.000000e+00 -0.0 -0.0
G08     1.007195e-02 -1.120548e-17 -1.970795e-17 ...   6.479663e-02 -0.0 -0.0
G09    -0.000000e+00 -0.000000e+00 -0.000000e+00 ...  -0.000000e+00  0.1 -0.0
G10    -0.000000e+00 -0.000000e+00 -0.000000e+00 ...  -0.000000e+00 -0.0  0.1

[10 rows x 10 columns],
'N': 10,
'created': {'creator': 'Nordlinglab _Justin',
'id': '58909',
'nodes': '10',
'sparsity': 14,
'time': '1555834954',
'type': 'random'},
'description': 'This is a sparse network with 10 nodes, 10 negative '
'self-loops and 15 randomly chosen links generated by Justin '
'2018-08-31. The coefficients are chosen such that they form '
'one strong component and a stable dynamical system with time '
'constants in the range 0.089 to 12 and an interampatteness '
'level of 145 that is in-between the estimated level of an E. '
'coli (Gardner et al. 2003 Science) and Yeast (Lorenz et al. '
'2009 PNAS) gene regulatory network. The coefficients of the '
'network have not been tuned to explain any of the data sets '
'in the mentioned articles.',
'name': 'Nordlinglab _Justin-D20190421-random-N10-L14ID58909',
'names': ['G01',
'G02',
'G03',
'G04',
'G05',
'G06',
'G07',
'G08',
'G09',
'G10'],
'nodes': 0    G01
1    G02
2    G03
3    G04
4    G05
5    G06
6    G07
7    G08
8    G09
9    G10
Name: node, dtype: object,
'params': source  target
G01     G01       -9.999993
G06      -46.352859
G08       34.951935
G02     G02       -9.999993
G03     G03       -9.999993
G04     G04       -9.999993
G05     G05       -9.999993
G06     G01       46.352859
G06       -9.999993
G07     G07       -9.999993
G08     G01      -34.951935
G08       -9.999993
G09     G09       -9.999993
G10     G10       -9.999993
Name: value, dtype: float64,
'shape': (10, 10),
'structure': 'random',
'tol': 2.220446049250313e-16}
```

### Data generation in JATNIpy

We use the network generated by JATNIpy to simulate perturbation experiments. We simulate N single gene perturbation experiments. Each gene is perturbed one after another followed by `N/2` experiments randomly.
```python=
from datastruct.Dataset import Dataset
Da = Dataset()
import numpy as np
import scipy
from scipy import sparse as sparse
from numpy import linalg as LA
from numpy import random as rd
SNR = 7
alpha = 0.01
c=sparse.rand(N,int(N/2),density=0.2,format='coo',dtype=None,random_state=None)
d=np.logical_and(c.A,1)
g = np.eye(N)
P=np.concatenate((g,d),axis = 1)
G = np.asarray(Net.G)
Y = np.dot(G,P)
s=LA.svd(Y, full_matrices=True)[1]
data = Da.Dataset()
stdE = s[N-1]/(((scipy.stats.chi2.ppf(1-alpha,P.shape[0]*P.shape[1]))**0.05)*SNR)
E = rd.rand(P.shape[0],P.shape[1])*stdE
F = np.zeros((P.shape[0],P.shape[1]))
```
We create a perturbation matrix `P` and a corresponding response matrix `Y`. The standard deviation is used to generate the noise matrix `E` with Signal Noise Ratio (`SNR`) equal to 7. We don't use the input noise `F` here, but we still should define it, therefore, we set it to zero. Then we populate a `Dataset` object with these information.

```python=21
D = Da.Dataset()
D.network = Net.name
import pandas as pd
names = pd.Series(Net.names, name="node")
names.name = "node"
D.M = P.shape[1]
M = D.M
D.N = P.shape[0]
N = D.N
samples = pd.Series(["S" + str(i + 1) for i in range(M)], name="sample")
D.E = pd.DataFrame(E, index=names, columns=samples)
D.F = pd.DataFrame(F, index=names, columns=samples)
D.Y = pd.DataFrame(Y+D.E, index=names, columns=samples)
D.P = pd.DataFrame(P, index=names, columns=samples)
D.lamda = pd.Series(np.array([np.array(i) for i in [stdE**2,0]]), index=["E", "F"])
sdY1 = pd.DataFrame(np.eye(int(D.N)), index=D.Y.index, columns=D.Y.index)
sdY2 = pd.DataFrame(pd.DataFrame(stdE*np.ones((D.P.shape[0],D.P.shape[1])), index=D.Y.index, columns=D.Y.columns), index=D.Y.index, columns=D.Y.columns)
frames1_Y = [sdY1,sdY2]
sdY_c_1 = pd.concat(frames1_Y,axis = 1)
sdY3 = pd.DataFrame(np.asarray(pd.DataFrame(stdE*np.ones((D.P.shape[0],D.P.shape[1])), index=D.Y.index, columns=D.Y.columns)).transpose(), index=D.Y.columns, columns=D.Y.index)
sdY4 = pd.DataFrame(np.eye(int(D.M)), index=D.Y.columns, columns=D.Y.columns)
frames2_Y = [sdY3,sdY4]
sdY_c_2 = pd.concat(frames2_Y,axis = 1)
frames_Y = [sdY_c_1,sdY_c_2]
D.E_covariance_element =  pd.concat(frames_Y,axis = 0)
D.E_covariance_variable = pd.DataFrame(D.lamda[0]*np.eye(N), index=names, columns=names)
D.F_covariance_variable = pd.DataFrame(np.zeros((N,N)), index=names, columns=names)
sdP1 = pd.DataFrame(np.eye(int(D.N)), index=D.Y.index, columns=D.Y.index)
sdP2 = pd.DataFrame(pd.DataFrame(np.zeros((D.P.shape[0],D.P.shape[1])), index=D.P.index, columns=D.P.columns), index=D.Y.index, columns=D.Y.columns)
frames1_P = [sdP1,sdP2]
sdP_c_1 = pd.concat(frames1_P,axis = 1)	
sdP3 = pd.DataFrame(np.asarray(pd.DataFrame(np.zeros((D.P.shape[0],D.P.shape[1])), index=D.P.index, columns=D.P.columns)).transpose(), index=D.Y.columns, columns=D.Y.index)
sdP4 = pd.DataFrame(np.eye(int(D.M)), index=D.Y.columns, columns=D.Y.columns)
frames2_P = [sdP3,sdP4]
sdP_c_2 = pd.concat(frames2_P,axis = 1)
frames_P = [sdP_c_1,sdP_c_2]
D.F_covariance_element = pd.concat(frames_P,axis = 0)
```
In order to initialize the `datastruct.Dataset` object with data, we then do the following code snippet:
```python=58
Data = Da.Dataset(D,Net)
creator_setting = {'creator':'Nordlinglab_Justin'}
Da.setname(Data,creator_setting)
Data.description = 'This data set contains 15 simulated experiments with additive white Gaussian noise with variance 0.00028 added to the response in order to make the SNR 7 and the data partly informative for network inference. The singular values of the response matrix are in the range 0.77 to 1.2.'
Data.nodes = Net.names
```
After initializing the `datastruct.Dataset` object, we then save it with the following command
```python=63
Dataset.save(Dataset,Data,'savepath/','.json')
```
Then, you can use these code to display the new `Dataset` object that you have generated in your personal folder. You should use your personal directory name instead of `Nordlinglab_Justin-ID17795-D20190415-N10-E15-SNR277998-IDY17795.json`
```python=64
Data = Dataset(Dataset.Load('./savepath/Nordlinglab_Justin-ID17795-D20190415-N10-E15-SNR277998-IDY17795.json'))
Data.__dict__
```
The displayed output is
* `E`: is expression noise matrix
* `E_covariance_element`: is measurement point variation of response
* `E_covariance_variable`: is covariance of noisy response
* `F`: is perturbation noise matrix
* `F_covariance_element`: is measurement point variation of perturbation
* `E_covariance_variable`: is covariance of pertubations
* `M`: is perturbations matrix
* `N`: is gene number
* `P`: is perturbations number
* `Y`: is response matrix
* `__model_eq__`: shows the model equation
* `created`: is the basic information of the network
* `nodes`: are names for each gene nodes
* `description`: is data set description
* `lamda`: includes expression and perturbation noise variable `E_variance` and `F_variance`
* `network`: is network name
* `nodes`: are names for each gene nodes

```
{'E': sample        S1        S2        S3    ...          S13       S14       S15
node                                    ...                                 
G01     0.001609  0.000334  0.000934    ...     0.000664  0.000593  0.000298
G02     0.001145  0.000808  0.000116    ...     0.000497  0.001746  0.001446
G03     0.001796  0.000210  0.000173    ...     0.000684  0.001074  0.001337
G04     0.000962  0.000706  0.001298    ...     0.000916  0.000908  0.000054
G05     0.001347  0.001081  0.000509    ...     0.001760  0.001839  0.000857
G06     0.001201  0.000180  0.001394    ...     0.000518  0.000246  0.001442
G07     0.001585  0.000752  0.000197    ...     0.000918  0.000703  0.000525
G08     0.001627  0.001223  0.001366    ...     0.001103  0.000204  0.001348
G09     0.001298  0.000705  0.000014    ...     0.000690  0.001257  0.000649
G10     0.000353  0.000890  0.001845    ...     0.001720  0.000523  0.000738

[10 rows x 15 columns],
'E_covariance_element':           G01       G02       G03    ...          S13       S14       S15
G01  1.000000  0.000000  0.000000    ...     0.001864  0.001864  0.001864
G02  0.000000  1.000000  0.000000    ...     0.001864  0.001864  0.001864
G03  0.000000  0.000000  1.000000    ...     0.001864  0.001864  0.001864
G04  0.000000  0.000000  0.000000    ...     0.001864  0.001864  0.001864
G05  0.000000  0.000000  0.000000    ...     0.001864  0.001864  0.001864
G06  0.000000  0.000000  0.000000    ...     0.001864  0.001864  0.001864
G07  0.000000  0.000000  0.000000    ...     0.001864  0.001864  0.001864
G08  0.000000  0.000000  0.000000    ...     0.001864  0.001864  0.001864
G09  0.000000  0.000000  0.000000    ...     0.001864  0.001864  0.001864
G10  0.000000  0.000000  0.000000    ...     0.001864  0.001864  0.001864
S1   0.001864  0.001864  0.001864    ...     0.000000  0.000000  0.000000
S2   0.001864  0.001864  0.001864    ...     0.000000  0.000000  0.000000
S3   0.001864  0.001864  0.001864    ...     0.000000  0.000000  0.000000
S4   0.001864  0.001864  0.001864    ...     0.000000  0.000000  0.000000
S5   0.001864  0.001864  0.001864    ...     0.000000  0.000000  0.000000
S6   0.001864  0.001864  0.001864    ...     0.000000  0.000000  0.000000
S7   0.001864  0.001864  0.001864    ...     0.000000  0.000000  0.000000
S8   0.001864  0.001864  0.001864    ...     0.000000  0.000000  0.000000
S9   0.001864  0.001864  0.001864    ...     0.000000  0.000000  0.000000
S10  0.001864  0.001864  0.001864    ...     0.000000  0.000000  0.000000
S11  0.001864  0.001864  0.001864    ...     0.000000  0.000000  0.000000
S12  0.001864  0.001864  0.001864    ...     0.000000  0.000000  0.000000
S13  0.001864  0.001864  0.001864    ...     1.000000  0.000000  0.000000
S14  0.001864  0.001864  0.001864    ...     0.000000  1.000000  0.000000
S15  0.001864  0.001864  0.001864    ...     0.000000  0.000000  1.000000

[25 rows x 25 columns],
'E_covariance_variable': node       G01       G02       G03    ...          G08       G09       G10
node                                  ...                                 
G01   0.000003  0.000000  0.000000    ...     0.000000  0.000000  0.000000
G02   0.000000  0.000003  0.000000    ...     0.000000  0.000000  0.000000
G03   0.000000  0.000000  0.000003    ...     0.000000  0.000000  0.000000
G04   0.000000  0.000000  0.000000    ...     0.000000  0.000000  0.000000
G05   0.000000  0.000000  0.000000    ...     0.000000  0.000000  0.000000
G06   0.000000  0.000000  0.000000    ...     0.000000  0.000000  0.000000
G07   0.000000  0.000000  0.000000    ...     0.000000  0.000000  0.000000
G08   0.000000  0.000000  0.000000    ...     0.000003  0.000000  0.000000
G09   0.000000  0.000000  0.000000    ...     0.000000  0.000003  0.000000
G10   0.000000  0.000000  0.000000    ...     0.000000  0.000000  0.000003

[10 rows x 10 columns],
'F': sample   S1   S2   S3   S4   S5   S6 ...   S10  S11  S12  S13  S14  S15
node                                 ...                               
G01     0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0
G02     0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0
G03     0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0
G04     0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0
G05     0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0
G06     0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0
G07     0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0
G08     0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0
G09     0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0
G10     0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0

[10 rows x 15 columns],
'F_covariance_element':      G01  G02  G03  G04  G05  G06  G07 ...    S9  S10  S11  S12  S13  S14  S15
G01  1.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
G02  0.0  1.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
G03  0.0  0.0  1.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
G04  0.0  0.0  0.0  1.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
G05  0.0  0.0  0.0  0.0  1.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
G06  0.0  0.0  0.0  0.0  0.0  1.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
G07  0.0  0.0  0.0  0.0  0.0  0.0  1.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
G08  0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
G09  0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
G10  0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
S1   0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
S2   0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
S3   0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
S4   0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
S5   0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
S6   0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
S7   0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
S8   0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  0.0
S9   0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   1.0  0.0  0.0  0.0  0.0  0.0  0.0
S10  0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  1.0  0.0  0.0  0.0  0.0  0.0
S11  0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  1.0  0.0  0.0  0.0  0.0
S12  0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  1.0  0.0  0.0  0.0
S13  0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  1.0  0.0  0.0
S14  0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  1.0  0.0
S15  0.0  0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0  1.0

[25 rows x 25 columns],
'F_covariance_variable': node  G01  G02  G03  G04  G05  G06  G07  G08  G09  G10
node                                                  
G01   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
G02   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
G03   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
G04   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
G05   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
G06   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
G07   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
G08   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
G09   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0
G10   0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0,
'M': 15,
'N': 10,
'P': sample   S1   S2   S3   S4   S5   S6 ...   S10  S11  S12  S13  S14  S15
node                                 ...                               
G01     1.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0
G02     0.0  1.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  1.0  0.0  0.0
G03     0.0  0.0  1.0  0.0  0.0  0.0 ...   0.0  0.0  0.0  1.0  1.0  0.0
G04     0.0  0.0  0.0  1.0  0.0  0.0 ...   0.0  0.0  0.0  0.0  1.0  0.0
G05     0.0  0.0  0.0  0.0  1.0  0.0 ...   0.0  0.0  0.0  0.0  0.0  0.0
G06     0.0  0.0  0.0  0.0  0.0  1.0 ...   0.0  0.0  0.0  0.0  1.0  0.0
G07     0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  1.0  0.0  1.0  0.0  0.0
G08     0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  0.0  1.0  0.0  0.0  0.0
G09     0.0  0.0  0.0  0.0  0.0  0.0 ...   0.0  1.0  0.0  0.0  0.0  0.0
G10     0.0  0.0  0.0  0.0  0.0  0.0 ...   1.0  0.0  0.0  0.0  1.0  0.0

[10 rows x 15 columns],
'Y': sample        S1        S2        S3    ...          S13       S14       S15
node                                    ...                                 
G01     0.004491  0.000334  0.000934    ...     0.000664  0.013951  0.000298
G02     0.001145  0.100809  0.000116    ...     0.100497  0.001746  0.001446
G03     0.001796  0.000210  0.100173    ...     0.100685  0.101074  0.001337
G04     0.000962  0.000706  0.001298    ...     0.000916  0.100908  0.000054
G05     0.001347  0.001081  0.000509    ...     0.001760  0.001839  0.000857
G06    -0.012157  0.000180  0.001394    ...     0.000518  0.038331  0.001442
G07     0.001585  0.000752  0.000197    ...     0.100918  0.000703  0.000525
G08     0.011699  0.001223  0.001366    ...     0.001103  0.046890  0.001348
G09     0.001298  0.000705  0.000014    ...     0.000690  0.001257  0.000649
G10     0.000353  0.000890  0.001845    ...     0.001720  0.100523  0.000738

[10 rows x 15 columns],
'__model_eq__': 'X ~ -dot(P, pinv(A).T)',
'created': {'creator': 'Nordlinglab_Justin',
'id': '58909',
'nodes': '10',
'sparsity': 14,
'time': '1555834954',
'type': 'random'},
'description': 'This data set contains 15 simulated experiments with additive '
'white Gaussian noise with variance 0.00028 added to the '
'response in order to make the SNR 7 and the data partly '
'informative for network inference. The singular values of the '
'response matrix are in the range 0.77 to 1.2.',
'lamda': E_variance    0.000003
F_variance    0.000000
dtype: float64,
'name': 'Nordlinglab_Justin-ID58909-D20190422-N10-E15-SNR655-IDY58909',
'network': 'Nordlinglab _Justin-D20190421-random-N10-L14ID58909',
'nodes': 0    G01
1    G02
2    G03
3    G04
4    G05
5    G06
6    G07
7    G08
8    G09
9    G10
Name: node, dtype: object,
'tol': 2.220446049250313e-16}
```
If the data is generated by yourself using the computer, then, it can connect a dataset to a network. So, the `network` is reported in the `Data` object.
We also provide normalisation procedures for the `Data` object that will normalize the expression matrix `Y`. Three different normalisation procedures are available:
1. Standard normalisation
1. Min/Max range scaling
1. Unit scaling. 


All methods works over rows or columns, depending on input. Some usage examples are:

1. For standard normalisation
```python=
NewData = Da.std_normalize(Data,2)
np.sum(Da.response(NewData),axis = 1)
np.sum(Da.response(NewData)**2,axis = 1)
```
should return zeros as sum over rows and the squared values should be 1 for each sample so the sum over rows should be = M.

2. For range scaling
```python=
NewData = Da.range_scaling(Data,2)
np.max(Da.response(NewData),axis=1)
np.min(Da.response(NewData),axis=1)
```   
the max and min of each row should be 1 and 0 respectively.

3. For unit scaling
```python=
NewData = Da.unit_length_scaling(Data,2)
np.sum(Da.response(NewData)**2,axis = 1)
```
the squared values should sum to 1.



It should be noted that the noise estimates are currently not scaled according to the new data and should therefore not be used *as is* in subsequent calculations.

### Save Dataset
```python=
Dataset.save(Dataset,Data,'savepath/','.json')
```

### Analyse

The `analyse` folder provides the programming to analyse networks, datasets and benchmark results.

We first demonstrate how to load the example network `Nordling-D20100302-random-N10-L25-ID1446937.json` and dataset `Nordling-ID1446937-D20150825-E15-SNR3291-IDY15968.json` from the online repository with the following command:
```python=
from datastruct.Dataset import Dataset
Data = Dataset(Dataset.Load('test'))
from datastruct.Network import Network
Net = Network(Network.Load('test'))
```

#### Network analysis in JATNIpy:

We input the `Net` to the `analyse.Model` to analyse the network:

```python=5
from analyse.Model import Model
M = Model()
net_prop = M.Model(Net)
net_prop.__dict__
```

It then produces the output like the following:
```
{
		'network': 'Nordling-D20100302-random-N10-L25-ID1446937',
		'interampatteness': 144.6936524435306,
		'NetworkComponents': 1,
		'AvgPathLength': 2.8777777777777778,
		'tauG': 0.08503206402335546
		'CC': 0.05,
		'DD': 1.5,
}
```
There are six measures be calculated in JATNIpy: 
* `interampatteness`: which is the condition number of `A` calculated by using `numpy.linalg.cond(A)`. 
* `NetworkComponents`: The number of strongly connected components,  is calculated by `graphconncomp` function.
* `AvgPathLength`: is the path length of the graph of the network uses `median_path_length` function. No matter the `graphconncomp` or `median_path_length` are used the python based `networkx` package to calculate.
* `tauG`: is the time constant of the network
* `CC`: is the average clustering coefficient that can be explained as as the neighborhood sparsity of each node in the network but not considering itself. 
* `DD`: is the average degree distribution of model. 

The property `analyse.Model.type` can be selected as `directed` or `undirected` by using `analyse.DataModel.type`.

Not only the average clustering coefficients but also all clustering coefficients can be calculated. All clustering coefficients can be calculated by
```python=9
CCs = M.clustering_coefficient(Net)
```
#### Data analysis in JATNIpy:

We input the `Data` to the `analyse.DataAnalysis` to analyse the data:
```python=
from analyse.DataAnalysis import DataAnalysis
DD = DataAnalysis()
data_prop = DD.Data(Data)
data_prop.__dict__
```

It then produces the output like the following:
```
{
'dataset': 'Nordling-ID1446937-D20150825-N10-E15-SNR3291-IDY15968',
'SNR\textunderscore Phi\textunderscore true': 6.999999962249559,
'SNR\textunderscore Phi\textunderscore gauss': 3.3098514156225645,
'SNR\textunderscore phi\textunderscore true': 10.991358740090298,
'SNR\textunderscore phi\textunderscore gauss': 10.340857240865667
}   
```
The following two functions calculates the SNR for all
```python=6
SNRe = DD.calc_SNR_phi_true(Data)
SNRl = DD.calc_SNR_phi_gauss(Data)
```
#### Performance evaluation in JATNIpy:

In order to analyze the performance of an inference method, we first need to generate an output. It is easy to manipulate by using wrappers. Each method has an associated wrapper that parses the data of the method itself.
In JATNIpy, we re-implement four wrappers which are LASSO, Glmnet, LSCO, RNICO, respectively.
Before running Glmnet implementation, we should do these command lines
```bash=
apt-get -y update
apt-get install -y libatlas-base-dev
apt-get install -y python3-tk
apt-get install libgfortran3
```
* To run the `Glmnet` method we execute:
```python=
from Methods.Glmnet import Glmnet
estA,zetavec,zetaRange = Glmnet(Data,'full')
```
to use Glmnet you should install first (if you install in virtual, then don't need this step)
```bash=
pip3 install python-glmnet
```
* To run the `lsco` implementation we execute:
```python=
from Methods.lsco import lsco
estA,zetavec,zetaRange = lsco(Data,'full')
```

* To run the `RNI` implementation we execute:
```python=
from Methods.RNI import RNI
estA,zetavec,zetaRange = RNI(Data,'full')
```

* To run the `Lasso` implementation we execute:
```python=
from Methods.Lasso import Lasso
estA,zetavec,zetaRange = Lasso(Data,'full')
```


The returned regularization parameters used within the algorithm is `zetavec`. The parameter `'full'` implies that the method will generate the complete regularization path from full to empty network with the `zeta` values scaled between 0 and 1. 
Only for the `RNI` method, a zetavec can be specified and supplied to it in JATNIpy. And `zetaRange` can scale the factor used for the parameters.  Then, RNI will use the vector of values to infer the networks

```python=
zetavec = np.logspace(-6,0,100)
estA = RNI(Data,zetavec)[0]
```
and the method will use that vector of values to infere the networks.

To analyse the performance of the model, we input the network estimates
produced by the algorithm to the model comparison method:

```python=
from analyse.CompareModels import CompareModels
import numpy as np
M = CompareModels()
M = M.set_A(M,Net)
M = M.CompareModels(M,np.asarray(Net.A),estA)
```

The `max` function in `CompareModels` is used to find the maximum for each calculated measure:

```python=6
maxM = M.max(M)
```

And the `maxM` contains the maximum of all measures in `CompareModels` . If you want to get the optimal performance for specific measure like `'MCC'`:

```python=7
maxM[0]['MCC']
```
### How to publish JATNIpy on Pypi ###
First, you should enter [Pypi_register](https://pypi.org/account/register/) and register a new account of Pypi.

Second, in order to upload your package to [PyPI](https://pypi.org/), you’ll use a tool called Twine. You can install Twine using Pip as usual:
```bash=
pip install twine
```

Packages on PyPI are not distributed as plain source code. They are wrapped into distribution packages. The most common formats for distribution packages are source archives and Python wheels.

In order to create a source archive and a wheel for your package, you can run the following command:
```bash=
python3 setup.py sdist
```

Before runing this command, you should check the setup.py, you can refer [Packaging Python Projects](https://packaging.python.org/tutorials/packaging-projects/). Remember to change the version. The newest version, should not have the same name as previous ones.

If you have your own package to publish, this final step is short:
(Check that there is only the newest version the dist folder.)
```bash=
twine upload dist/*
```
And you should input your account and password to update the version 



### Who do I talk to? ###

* For questions contact [Justin](mailto:justin.lin@nordlinglab.org), [Andreas Tjärnberg](mailto:at145@nyu.edu) 
, [Torbjörn Nordling](mailto:t@nordlinglab.org)



