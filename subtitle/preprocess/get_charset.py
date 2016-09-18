import chardet
import sys
import os

if __name__ == '__main__':
    for dir in ("srt", "ass", "lrc", "ssa", "str", "vtt"):
        for root, dirs, files in os.walk(dir):
            for file in files:
                file_path = root + "/" + file
                f = open(file_path,'r')
                data = f.read()
                encoding = chardet.detect(data)["encoding"]
                if encoding not in ("UTF-8-SIG", "UTF-16LE", "utf-8"):
                    print file_path, encoding
                f.close()
