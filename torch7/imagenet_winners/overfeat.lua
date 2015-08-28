function overfeat_fast(lib)
   local SpatialConvolution = lib[1]
   local SpatialMaxPooling = lib[2]
   local ReLU = lib[3]
   local SpatialZeroPadding = nn.SpatialZeroPadding

   local features = nn.Sequential()
   features:add(SpatialConvolution(3, 96, 11, 11, 4, 4))
   features:add(ReLU(true))
   features:add(SpatialMaxPooling(2, 2, 2, 2))
   features:add(SpatialConvolution(96, 256, 5, 5, 1, 1))
   features:add(ReLU(true))
   features:add(SpatialMaxPooling(2, 2, 2, 2))
   features:add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
   features:add(ReLU(true))
   features:add(SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1))
   features:add(ReLU(true))
   features:add(SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
   features:add(ReLU(true))
   features:add(nn.SpatialMaxPooling(2, 2, 2, 2))

   features:get(1).gradInput = nil

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
