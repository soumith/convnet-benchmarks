include ccv/lib/config.mk

LDFLAGS := -L"ccv/lib" -lccv $(LDFLAGS)
CFLAGS := -O3 -Wall -I"ccv//lib" $(CFLAGS)
NVFLAGS := -O3 -I"ccv/lib" -lineinfo $(NVFLAGS)

all: libccv.a cwc-bench 

clean:
	${MAKE} clean -C ccv/lib ; rm -f *.o $(TARGETS)

cwc-bench: %: %.o cwc-bench-runtime.o libccv.a
	$(CC) -o $@ cwc-bench-runtime.o $< $(LDFLAGS)

libccv.a:
	${MAKE} -C ccv/lib

%.o: %.c ccv/lib/ccv.h
	$(CC) $< -o $@ -c $(CFLAGS)

%.o: %.cu ccv/lib/ccv.h ccv/lib/cuda/*.h ccv/lib/cuda/*.cu
	$(NVCC) $< -o $@ -c $(NVFLAGS)
