#!/bin/bash

PROGLOC="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CURLOC=`pwd`

cd $PROGLOC

rm -rf __pycache__
rm -rf */__pycache__
rm -rf */*/__pycache__
rm -rf */*/*/__pycache__
rm -rf */*/*/*/__pycache__
rm -rf */*/*/*/*/__pycache__
rm -rf .idea
rm -rf */.idea
rm -rf */*/.idea
rm -rf */*/*/.idea
rm -rf */*/*/*/.idea
rm -rf */*/*/*/*/.idea

cd $CURLOC