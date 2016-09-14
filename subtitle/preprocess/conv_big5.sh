#!/bin/bash

while read line
do
    file=`echo $line|awk '{print $1}'`; echo $file;
    iconv -f big5 -t utf-8 $file > ${file}.2
    if [ $? -eq 0 ];then
        mv ${file}.2 ${file}
    else
        rm ${file}.2
    fi
done
