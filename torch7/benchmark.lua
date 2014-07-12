require 'nn'
require 'xlua'
require 'pl'
 
 
opt = lapp[[
   -t,--threads            (default 6)           number of threads
   -p,--type               (default float)       float or cuda
   -i,--devid              (default 1)           device ID (if using CUDA)
]]
 
p = xlua.Profiler()
torch.setnumthreads(opt.threads)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')
 
if opt.type == 'cuda' then
   require 'cunn'
   cutorch.setDevice(opt.devid)
   print('==> using GPU #' .. cutorch.getDevice())
end
 
batchSize = 128
iH = 128
iW = 128
fin = 3
fout = 96
kH = 11
kW = 11

if opt.type == 'cuda' then   
   input = torch.CudaTensor(fin,iH,iW,batchSize)
   model = nn.SpatialConvolutionCUDA(fin, fout, kH, kW, 1, 1):cuda()
else
   input = torch.FloatTensor(batchSize,fin,iH,iW) 
   model = nn.SpatialConvolution(fin, fout, kH, kW)   
end
p:start('spatialconv')
output = model:forward(input)
if opt.type == 'cuda' then cutorch.synchronize() end
p:lap('spatialconv')
p:printAll{}
 
 
print('Gops/s:', ( batchSize*fin*fout*kH*kW*((iH-kH)+1)*((iW-kW)+1)*2 ) / p:cpu('spatialconv') / 1e9 ) -- 2 operations MUL, ACC
