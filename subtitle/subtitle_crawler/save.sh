#! /bin/bash 
###########################################
# Save file with CURL
###########################################

# constants
baseDir=$(cd `dirname "$0"`;pwd)
SAVE_PATH=$baseDir/result
# functions

# main 
[ -z "${BASH_SOURCE[0]}" -o "${BASH_SOURCE[0]}" = "$0" ] || return
if [ ! -d "$SAVE_PATH" ]; then
  # Control will enter here if $DIRECTORY doesn't exist.
  mkdir -p $SAVE_PATH
fi

cd $SAVE_PATH
curl -JLO $1
