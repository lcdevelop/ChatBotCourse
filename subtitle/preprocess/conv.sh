#!/bin/bash

while read line
do
    file=`echo $line|awk '{print $1}'`
    iconv -f gb2312 -t utf-8 $file > ${file}.2 2>/dev/null
    if [ $? -eq 0 ];then
        mv ${file}.2 ${file}
        echo "mv ${file}.2 ${file}"
    else
        rm ${file}.2
    fi
done
