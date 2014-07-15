
Clone nnForge
-------------
```
git clone https://github.com/milakov/nnForge.git
cd nnForge
```

Modify Settings.mk
------------------
```
diff --git a/Settings.mk b/Settings.mk
index 3b8b945..96f030f 100644
--- a/Settings.mk
+++ b/Settings.mk
@@ -1,20 +1,20 @@
 BUILD_MODE=release
 ENABLE_CUDA_BACKEND=yes
-ENABLE_CUDA_PROFILING=no
+ENABLE_CUDA_PROFILING=yes
 CPP11COMPILER=no
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

-BOOST_LIBS=-lboost_thread-mt -lboost_regex-mt -lboost_chrono-mt -lboost_filesystem-mt -lboost_program_options-mt -lboost_random-mt -lboost_system-mt -lboost_date_time-mt
+BOOST_LIBS=-lboost_thread -lboost_regex -lboost_chrono -lboost_filesystem -lboost_program_options -lboost_random -lboost_system -lboost_date_time
 OPENCV_LIBS=-lopencv_highgui -lopencv_imgproc -lopencv_core
 NETCDF_LIBS=-lnetcdf
 MATIO_LIBS=-lmatio
```

Compile nnForge
---------------
```
./make_all.sh
```
