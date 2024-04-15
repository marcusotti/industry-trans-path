import pyomo.environ as py
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import math
from pyomo.environ import Param
import os
import pandas as pd
from pandas import ExcelWriter

def print_param(param, *args):
    dim = len(args)

    if dim == 1 or dim == 2:
        if dim == 1:
            data = pd.DataFrame(index=args[0], columns=['value'])
            for x in args[0]:
                data.loc[x, 'value'] = param[x]
        elif dim == 2:
                data = pd.DataFrame(index=args[0], columns=args[1])
                for x in args[0]:
                    for y in args[1]:
                        data.loc[x, y] = param[x, y]
        print('##############################################################')
        print(param)
        print(data)
        print('##############################################################')
    elif dim == 3:
        print('##############################################################')
        print(param)
        for x in args[0]:
            data = pd.DataFrame(index=args[1], columns=args[2])
            print(x)
            for y in args[1]:
                for z in args[2]:
                    data.loc[y, z] = param[y, z]
            print(data)
        print('##############################################################')
    return

def print_var(var, *args):
    dim = len(args)

    if dim == 1 or dim == 2:
        if dim == 1:
            data = pd.DataFrame(index=args[0], columns=['value'])
            for x in args[0]:
                data.loc[x, 'value'] = var[x].value
        elif dim == 2:
                data = pd.DataFrame(index=args[0], columns=args[1])
                for x in args[0]:
                    for y in args[1]:
                        data.loc[x, y] = var[x, y].value
        print('##############################################################')
        print(var)
        print(data)
        print('##############################################################')
    elif dim == 3:
        print('##############################################################')
        print(var)
        for x in args[0]:
            data = pd.DataFrame(index=args[1], columns=args[2])
            print(x)
            for y in args[1]:
                for z in args[2]:
                    data.loc[y, z] = var[x, y, z].value
                    # print(f'[{x} {y} {z}]: {var[x, y, z].value}')
            print(data)
        print('##############################################################')
    return

def var2excel(var, dir, *args):
    _results_dir = os.path.join(os.path.dirname(__file__), 'results')
    _results_dir = os.path.join(_results_dir, dir['name'])
    os.makedirs(_results_dir, exist_ok=True)

    dim = len(args)

    if dim == 2:
        data = pd.DataFrame(index=args[0], columns=args[1])
        for x in args[0]:
            for y in args[1]:
                data.loc[x, y] = var[x, y].value
        data.to_excel(os.path.join(_results_dir, str(var) + '.xlsx'), 
            sheet_name='data', index=True)

    elif dim == 3:
        with ExcelWriter(os.path.join(_results_dir, str(var) + '.xlsx')) as writer:
            for x in args[0]:
                data = pd.DataFrame(index=args[1], columns=args[2])
                for y in args[1]:
                    for z in args[2]:
                        data.loc[y, z] = var[x, y, z].value
                data.to_excel(writer, sheet_name=str(x), index=True)
    return