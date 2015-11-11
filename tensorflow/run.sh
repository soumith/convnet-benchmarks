export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib:/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/Downloads/cudnn-6.5-linux-x64-v2:$LD_LIBRARY_PATH

python benchmark_alexnet.py   2>&1 | tee output_alexnet.log
python benchmark_overfeat.py  2>&1 | tee output_overfeat.log
python benchmark_vgg.py       2>&1 | tee output_vgga.log
python benchmark_googlenet.py 2>&1 | tee output_googlenet.log
