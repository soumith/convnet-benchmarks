
Clone nnForge
-------------
```
git clone https://github.com/milakov/nnForge.git
cd nnForge
```

Modify Settings.mk
------------------
```
--- a/Settings.mk
+++ b/Settings.mk
@@ -1,18 +1,18 @@
 BUILD_MODE=release
 ENABLE_CUDA_BACKEND=yes
-ENABLE_CUDA_PROFILING=no
-CPP11COMPILER=no
+ENABLE_CUDA_PROFILING=yes
+CPP11COMPILER=yes
-BOOST_PATH=/usr/local
+BOOST_PATH=/usr/
 OPENCV_PATH=/usr/local
-NETCDF_INSTALLED=yes
+NETCDF_INSTALLED=no
 NETCDF_PATH=
-MATIO_INSTALLED=yes
+MATIO_INSTALLED=no
 MATIO_PATH=
 CUDA_PATH=/usr/local/cuda
 NVCC=nvcc
 NNFORGE_PATH=../..
-NNFORGE_INPUT_DATA_PATH=/home/max/nnforge/input_data
-NNFORGE_WORKING_DATA_PATH=/home/max/nnforge/working_data
+NNFORGE_INPUT_DATA_PATH=./nnforge/input_data
+NNFORGE_WORKING_DATA_PATH=./nnforge/working_data
```

Compile nnForge
---------------
```
./make_all.sh
```
