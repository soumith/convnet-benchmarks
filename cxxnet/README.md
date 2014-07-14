git clone https://github.com/antinucleon/cxxnet.git
cd cxxnet

Apply the diff:
```
diff --git a/Makefile b/Makefile
index 0eaf612..6dd13cd 100644
--- a/Makefile
+++ b/Makefile
@@ -3,7 +3,7 @@ export CC  = gcc
 export CXX = g++
 export NVCC =nvcc

-export CFLAGS = -Wall -g -O3 -msse3 -Wno-unknown-pragmas -funroll-loops -I./mshadow/
+export CFLAGS = -Wall -g -O3 -msse3 -Wno-unknown-pragmas -funroll-loops -I./mshadow/  -I /usr/local/cuda/include/ -L /usr/local/cuda/lib64/ -pthread

 ifeq ($(blas),1)
  LDFLAGS= -lm -lcudart -lcublas -lcurand -lz `pkg-config --libs opencv` -lblas
```

bash build.sh blas=1

