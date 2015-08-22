Download and extract viennacl:
wget -c http://downloads.sourceforge.net/project/viennacl/1.7.x/ViennaCL-1.7.0.tar.gz?r=http%3A%2F%2Fviennacl.sourceforge.net%2Fviennacl-download.html&ts=1440274908&use_mirror=superb-dca2
tar -xvf ViennaCL-1.7.0.tar.gz


Install Caffe using the script:
```bash
bash install.sh
```
(If something fails, check your ViennaCL (>= 1.5) and OpenCL installation. Might have issues with libOpenCL.so provided by nVidia. If so, install a second OpenCL implementation from Intel or AMD.)

Run the benchmark using:
```bash
./run_imagenet.sh
./run_nogradinput.sh
./run_forcegradinput.sh
```

Requires at least one device with a valid OpenCL driver installed.
The default compile settings (Makefile.config) do:
- Use ViennaCL for BLAS calls.
- Disable the CUDA backend.

