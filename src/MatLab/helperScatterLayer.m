classdef helperScatterLayer < nnet.layer.Layer
    %#codegen   
    properties     
        Shape
    end
      
    methods
        function this = helperScatterLayer(numInputs, name, shape)                        
            this.NumInputs = numInputs;            
            this.Name = name;
            this.Shape = shape;
        end

        function Z = predict(this, varargin)
                if numel(varargin) ~= 2
                    assert(0);
                end

                X = varargin;
                input = squeeze(X{1});
                pillarIdx = (X{2});
                [P,C,N] = size(input);
               
                a = zeros([this.Shape C N],'like',input);
                a_r = reshape(a,prod(this.Shape),C,N);
                
                rowIdx = [1:P]';

                for j = 1:N
                    indices = pillarIdx(:,1,j)>=1;
                    indices = rowIdx(indices);
                    maps = pillarIdx(indices,:,j)-1;
                    convertmaps = maps(:,2)*this.Shape(1)+maps(:,1)+1;
                    a_r(convertmaps,:,j) = input(indices,:,j);
                end
                Z = reshape(a_r, [this.Shape C N]);
        end
                
    end
end