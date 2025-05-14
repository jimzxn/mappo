import importlib
import importlib.machinery
import os.path as osp

def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return importlib.machinery.SourceFileLoader('', pathname).load_module()



