Install DeepCL:

```bash
bash install.sh
```

Launch the script:

```bash
( LD_LIBRARY_PATH=$PWD/DeepCL/python PYTHONPATH=$PWD/DeepCL/python python deepcl_benchmark.py 1 )
( LD_LIBRARY_PATH=$PWD/DeepCL/python PYTHONPATH=$PWD/DeepCL/python python deepcl_benchmark.py 2 )
( LD_LIBRARY_PATH=$PWD/DeepCL/python PYTHONPATH=$PWD/DeepCL/python python deepcl_benchmark.py 3 )
( LD_LIBRARY_PATH=$PWD/DeepCL/python PYTHONPATH=$PWD/DeepCL/python python deepcl_benchmark.py 4 )
( LD_LIBRARY_PATH=$PWD/DeepCL/python PYTHONPATH=$PWD/DeepCL/python python deepcl_benchmark.py 5 )
```

Results should appear in the `results.txt` file

