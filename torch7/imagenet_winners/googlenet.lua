-- adapted from nagadomi's CIFAR attempt: https://github.com/nagadomi/kaggle-cifar10-torch7/blob/cuda-convnet2/inception_model.lua
local function inception(depth_dim, input_size, config, lib)
   local SpatialConvolution = lib[1]
   local SpatialMaxPooling = lib[2]
   local ReLU = lib[3]

   local depth_concat = nn.Concat(depth_dim)
   local conv1 = nn.Sequential()
   conv1:add(SpatialConvolution(input_size, config[1][1], 1, 1)):add(ReLU(true))
   depth_concat:add(conv1)

   local conv3 = nn.Sequential()
   conv3:add(SpatialConvolution(input_size, config[2][1], 1, 1)):add(ReLU(true))
   conv3:add(SpatialConvolution(config[2][1], config[2][2], 3, 3, 1, 1, 1, 1)):add(ReLU(true))
   depth_concat:add(conv3)

   local conv5 = nn.Sequential()
   conv5:add(SpatialConvolution(input_size, config[3][1], 1, 1)):add(ReLU(true))
   conv5:add(SpatialConvolution(config[3][1], config[3][2], 5, 5, 1, 1, 2, 2)):add(ReLU(true))
   depth_concat:add(conv5)

   local pool = nn.Sequential()
   pool:add(SpatialMaxPooling(config[4][1], config[4][1], 1, 1, 1, 1))
   pool:add(SpatialConvolution(input_size, config[4][2], 1, 1)):add(ReLU(true))
   depth_concat:add(pool)

   return depth_concat
end

local function googlenet(lib)
   local SpatialConvolution = lib[1]
   local SpatialMaxPooling = lib[2]
   local SpatialAveragePooling = torch.type(lib[2]) == 'nn.SpatialMaxPooling' and nn.SpatialAveragePooling or cudnn.SpatialAveragePooling
   local ReLU = lib[3]
   local model = nn.Sequential()
   model:add(SpatialConvolution(3,64,7,7,2,2,3,3)):add(ReLU(true))
   model:add(SpatialMaxPooling(3,3,2,2,1,1))
   -- LRN (not added for now)
   model:add(SpatialConvolution(64,64,1,1,1,1,0,0)):add(ReLU(true))
   model:add(SpatialConvolution(64,192,3,3,1,1,1,1)):add(ReLU(true))
   -- LRN (not added for now)
   model:add(SpatialMaxPooling(3,3,2,2,1,1))
   model:add(inception(2, 192, {{ 64}, { 96,128}, {16, 32}, {3, 32}},lib)) -- 256
   model:add(inception(2, 256, {{128}, {128,192}, {32, 96}, {3, 64}},lib)) -- 480
   model:add(SpatialMaxPooling(3,3,2,2,1,1))
   model:add(inception(2, 480, {{192}, { 96,208}, {16, 48}, {3, 64}},lib)) -- 4(a)
   model:add(inception(2, 512, {{160}, {112,224}, {24, 64}, {3, 64}},lib)) -- 4(b)
   model:add(inception(2, 512, {{128}, {128,256}, {24, 64}, {3, 64}},lib)) -- 4(c)
   model:add(inception(2, 512, {{112}, {144,288}, {32, 64}, {3, 64}},lib)) -- 4(d)
   model:add(inception(2, 528, {{256}, {160,320}, {32,128}, {3,128}},lib)) -- 4(e) (14x14x832)
   model:add(SpatialMaxPooling(3,3,2,2,1,1))
   model:add(inception(2, 832, {{256}, {160,320}, {32,128}, {3,128}},lib)) -- 5(a)
   model:add(inception(2, 832, {{384}, {192,384}, {48,128}, {3,128}},lib)) -- 5(b)
   model:add(SpatialAveragePooling(7,7,1,1))
   model:add(nn.View(1024):setNumInputDims(3))
   -- model:add(nn.Dropout(0.4))
   model:add(nn.Linear(1024,1000)):add(nn.ReLU(true))
   -- model:add(nn.LogSoftMax())
   model:get(1).gradInput = nil
   return model,'GoogleNet', {128,3,224,224}
end

return googlenet
