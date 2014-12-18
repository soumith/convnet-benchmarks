function alexnet(lib)
   local SpatialConvolution = lib[1]
   local SpatialMaxPooling = lib[2]
   local ReLU = lib[3]

   local features = nn.Sequential() -- branch 1
   features:add(SpatialConvolution(3,64,11,4,0,1,4))       -- 224 -> 55
   features:add(ReLU())
   features:add(SpatialMaxPooling(3,2))                   -- 55 ->  27
   features:add(SpatialConvolution(64,192,5,1,2,1,3))       --  27 -> 27
   features:add(ReLU())
   features:add(SpatialMaxPooling(3,2))                   --  27 ->  13
   features:add(SpatialConvolution(192,384,3,1,1,1,3))      --  13 ->  13
   features:add(ReLU())
   features:add(SpatialConvolution(384,256,3,1,1,1,3))      --  13 ->  13
   features:add(ReLU())
   features:add(SpatialConvolution(256,256,3,1,1,1,3))      --  13 ->  13
   features:add(ReLU())
   features:add(SpatialMaxPooling(3,2))                   -- 13 -> 6

   local classifier = nn.Sequential()
   classifier:add(nn.Transpose({4,1},{4,2},{4,3}))
   classifier:add(nn.View(256*6*6):setNumInputDims(3))   
   -- classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   -- classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(4096, 1000))
   -- classifier:add(nn.LogSoftMax())

   local model = nn.Sequential()
   model:add(features):add(classifier)
   
   return model,'AlexNet',{128,3,224,224}
end

return alexnet
