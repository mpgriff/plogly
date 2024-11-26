# plog
Plog is a flexible tool for Plotting LOGs.

## Setup
To set up Plog, run from one directory above plog in the command line: 
```shell
pip install -e plog
```
this allows modification of the repository. Alternatively one may install directly from git using the command,
```shell
pip install git+https://github.com/tangentlabs/django-oscar-paypal.git
```

Plog is tested with Python 3.11.

## Usage
An example script is provided (plotlogs.py).  Simply change the directory and file prefix to match.  A minimal example is provided below for plotting Dart data.

```python
python
from plog import Dart
import matplotlib.pyplot as plt

bh = Dart('./Results/TSB1_23-Nov-2024/TSB1_23-Nov-2024')
bh.plot()
plt.savefig('./Plots/TSB1p1Y.png',dpi=300)
plt.show()
plt.close()
```
