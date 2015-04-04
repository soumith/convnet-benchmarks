#!/bin/bash

git clone --recursive https://github.com/hughperkins/DeepCL.git
( cd DeepCL/python; python setup.py build_ext -i )

