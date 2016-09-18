import chardet
import sys
import os

if __name__ == '__main__':
    if len(sys.argv) == 2:
        for root, dirs, files in os.walk(sys.argv[1]):
            for file in files:
                file_path = root + "/" + file
                f = open(file_path,'r')
                data = f.read()
                f.close()
                encoding = chardet.detect(data)["encoding"]
                if encoding not in ("UTF-8-SIG", "UTF-16LE", "utf-8", "ascii"):
                    try:
                        gb_content = data.decode("gb18030")
                        gb_content.encode('utf-8')
                        f = open(file_path, 'w')
                        f.write(gb_content.encode('utf-8'))
                        f.close()
                    except:
                        print "except:", file_path
