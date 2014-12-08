require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'ccn2'

--cutorch.setDevice(4)

fSize = {3, 96, 128, 128, 384}
inputSize = {3, 64, 64}
batchSize = 128

nettype = 'cudnn'

if nettype == 'cudnn' then
  model = nn.Sequential()
  model:add(cudnn.SpatialConvolution(fSize[1], fSize[2], 9, 9))
  model:add(cudnn.ReLU())
  model:add(cudnn.SpatialMaxPooling(2,2,2,2))
  model:add(cudnn.SpatialConvolution(fSize[2], fSize[3], 5, 5))
  model:add(cudnn.ReLU())
  model:add(cudnn.SpatialMaxPooling(2,2,2,2))
  model:add(cudnn.SpatialConvolution(fSize[3], fSize[4], 4, 4))
  model:add(cudnn.ReLU())
  model:add(cudnn.SpatialConvolution(fSize[4], fSize[5], 3, 3))
  model:add(cudnn.ReLU())
  model:add(cudnn.SpatialMaxPooling(2,2,2,2))
  model:add(cudnn.SpatialConvolution(fSize[5], fSize[5], 3, 3))
  --model:add(nn.Reshape(fSize[5]))
  --model:add(nn.Linear(fSize[5],1))
elseif nettype == 'ccn2' then
  model = nn.Sequential()
  model:add(nn.Transpose({1,4},{1,3},{1,2}))
  model:add(ccn2.SpatialConvolution(fSize[1], fSize[2], 9))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialMaxPooling(2,2))
  model:add(ccn2.SpatialConvolution(fSize[2], fSize[3], 5))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialMaxPooling(2,2))
  model:add(ccn2.SpatialConvolution(fSize[3], fSize[4], 5))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialConvolution(fSize[4], fSize[5], 3))
  model:add(nn.ReLU())
  model:add(ccn2.SpatialMaxPooling(2,2))
  model:add(ccn2.SpatialConvolution(fSize[5], fSize[5], 3))
elseif nettype == 'MM' then
  model = nn.Sequential()
  model:add(nn.SpatialConvolutionMM(fSize[1], fSize[2], 9, 9))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  model:add(nn.SpatialConvolutionMM(fSize[2], fSize[3], 5, 5))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  model:add(nn.SpatialConvolutionMM(fSize[3], fSize[4], 4, 4))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolutionMM(fSize[4], fSize[5], 3, 3))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2,2,2,2))
  model:add(nn.SpatialConvolutionMM(fSize[5], fSize[5], 3, 3))
  model:add(nn.Reshape(fSize[5]))
end

model = model:cuda()

input = torch.rand(batchSize, inputSize[1], inputSize[2], inputSize[3]):cuda()

-- first run
--print(model:forward(input):size())
output = model:forward(input)
cutorch.synchronize()

a = torch.Timer()
output = model:forward(input)
print('FORWARD free run time:', a:time().real)

cutorch.synchronize()
a:reset()
output = model:forward(input)
cutorch.synchronize()
print('FORWARD sync time:', a:time().real)

cutorch.synchronize()
a:reset()
model:backward(input, output)
print('BACKWARD free run time:', a:time().real)

cutorch.synchronize()
a:reset()
model:backward(input, output)
cutorch.synchronize()
print('BACKWARD sync time:', a:time().real)
