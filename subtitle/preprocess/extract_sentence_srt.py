# coding:utf-8
import chardet
import os
import re

cn=ur"([\u4e00-\u9fa5]+)"
pattern_cn = re.compile(cn)
jp1=ur"([\u3040-\u309F]+)"
pattern_jp1 = re.compile(jp1)
jp2=ur"([\u30A0-\u30FF]+)"
pattern_jp2 = re.compile(jp2)

for root, dirs, files in os.walk("./srt"):
    file_count = len(files)
    if file_count > 0:
        for index, file in enumerate(files):
            f = open(root + "/" + file, "r")
            content = f.read()
            f.close()
            encoding = chardet.detect(content)["encoding"]
            try:
                for sentence in content.decode(encoding).split('\n'):
                    if len(sentence) > 0:
                        match_cn =  pattern_cn.findall(sentence)
                        match_jp1 =  pattern_jp1.findall(sentence)
                        match_jp2 =  pattern_jp2.findall(sentence)
                        sentence = sentence.strip()
                        if len(match_cn)>0 and len(match_jp1)==0 and len(match_jp2) == 0 and len(sentence)>1 and len(sentence.split(' ')) < 10:
                            print sentence.encode('utf-8')
            except:
                continue
