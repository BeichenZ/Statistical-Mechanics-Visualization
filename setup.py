import conda.cli
import importlib.util
import sys

import pdb
pdb.set_trace()

# List of Required packages
requiredPackages = ['numpy', 
        'matplotlib',
        'scipy',
        'scrapy']


toBeInstalledPkgs = []

for pkg in requiredPackages:

    spec = importlib.util.find_spec(pkg)
    if spec is None:
        print('ínstalling package:'+pkg)
        conda.cli.main('conda', 'install',  '-y', 'numpy')

print("Set up finished")
