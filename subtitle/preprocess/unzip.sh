#!/bin/bash

i=0; for file in `ls`; do mkdir output/${i}; echo "unzip $file -d output/${i}";unzip -P abc $file -d output/${i} > /dev/null; ((i++)); done
i=0; for file in `ls`; do mkdir output/${i}; echo "${i} unrar x $file output/${i}";unrar x $file output/${i} > /dev/null; ((i++)); done
