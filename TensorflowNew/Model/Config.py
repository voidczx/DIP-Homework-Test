# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/10/1017:21
# File：  Config.py
# Engine：PyCharm

import configparser as cp

def GetConfig(configPath='./Config.ini'):
    parser = cp.ConfigParser()
    parser.read(configPath)
    configInts = [[key, int(value)] for key, value in parser.items('ints')]
    configFloats = [(key, float(value)) for key, value in parser.items('floats')]
    configStrings = [(key, str(value)) for key, value in parser.items('strings')]
    return dict(configInts + configFloats + configStrings)
