# coding:utf-8
from langconv import *
import sys

def tradition2simple(line):
    line = Converter('zh-hans').convert(line.decode('utf-8'))
    line = line.encode('utf-8')
    return line

if __name__ == '__main__':
    if len(sys.argv) == 2:
        f = open(sys.argv[1], "r")
        while True:
            line = f.readline()
            if line:
                print tradition2simple(line).strip()
            else:
                break
        f.close()
