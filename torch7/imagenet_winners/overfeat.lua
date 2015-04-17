function overfeat_fast(lib)
   local SpatialConvolution = lib[1]
   local SpatialMaxPooling = lib[2]
   local ReLU = lib[3]
   local SpatialZeroPadding = nn.SpatialZeroPadding
   local padding = true
   local stride1only = false
   if lib[5] == 'fbfft' then
      padding = false -- fbfft does not support implicit zero padding
      stride1only = true -- fbfft does not support convolutions that are not stride-1
   end

   local features = nn.Sequential()
   if stride1only then
      features:add(cudnn.SpatialConvolution(3, 96, 11, 11, 4, 4))
   else
      features:add(SpatialConvolution(3, 96, 11, 11, 4, 4))
   end
   features:add(ReLU(true))
   features:add(SpatialMaxPooling(2, 2, 2, 2))
   features:add(SpatialConvolution(96, 256, 5, 5, 1, 1))
   features:add(ReLU(true))
   features:add(SpatialMaxPooling(2, 2, 2, 2))
   if not padding then
      features:add(SpatialZeroPadding(1,1,1,1))
      features:add(SpatialConvolution(256, 512, 3, 3, 1, 1))
   else
      features:add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
   end
   features:add(ReLU(true))
   if not padding then
      features:add(SpatialZeroPadding(1,1,1,1))
      features:add(SpatialConvolution(512, 1024, 3, 3, 1, 1))
   else
      features:add(SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1))
   end
   features:add(ReLU(true))
   if not padding then
      features:add(SpatialZeroPadding(1,1,1,1))
      features:add(SpatialConvolution(1024, 1024, 3, 3, 1, 1))
   else
      features:add(SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
   end
   features:add(ReLU(true))
   -- hardcode this as cudnn. because https://github.com/torch/cunn/issues/53 . 
   -- It's a small pooling, it is insignificant in time for the benchmarks.
   features:add(cudnn.SpatialMaxPooling(2, 2, 2, 2)) 

   local classifier = nn.Sequential()
   classifier:add(nn.View(1024*6*6))
   -- classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(1024*6*6, 3072))
   classifier:add(nn.Threshold(0, 1e-6))
   -- classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(3072, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(4096, 1000))
   -- classifier:add(nn.LogSoftMax())

   local model = nn.Sequential()
   model:add(features):add(classifier)
   
   return model,'OverFeat[fast]',{128,3,231,231}
end

return overfeat_fast
