classdef batchSampler < handle
    
    properties
        sampledList;
        idx;
        name;
        exampleNum;
        indices;
    end
    
    methods
      
        function obj = batchSampler(sampleList, name)
          obj.sampledList = sampleList;
          obj.name = name;
          obj.exampleNum = numel(sampleList);
          obj.indices = randperm(obj.exampleNum);
          obj.idx = 1;
      end
      
      function samples = sample(obj,num)
          if (obj.idx + num >= obj.exampleNum)
              ret = obj.indices(obj.idx:obj.exampleNum);
              obj.reset();
          else
              ret = obj.indices(obj.idx:obj.idx+num-1);
              obj.idx = obj.idx + num;
          end
          
          samples = [];
          for i = 1:numel(ret)
              samples = [samples;obj.sampledList{ret(i)}];
          end
      end
      
      function obj = reset(obj)
          obj.indices = obj.indices(randperm(obj.exampleNum));
          obj.idx = 1;
      end
      
    end
    
end
