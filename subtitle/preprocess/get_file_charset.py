import chardet
import sys
import os

if __name__ == '__main__':
    if len(sys.argv) == 2:
        file_path = sys.argv[1]
        f = open(file_path,'r')
        data = f.read()
        encoding = chardet.detect(data)["encoding"]
        if encoding not in ("UTF-8-SIG", "UTF-16LE", "utf-8"):
            print file_path, encoding
        f.close()
