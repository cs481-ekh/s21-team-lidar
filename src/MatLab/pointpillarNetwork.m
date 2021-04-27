function lgraph = pointpillarNetwork(numAnchors,gridParams,P,N,numClasses)
% This function is used to create the Pointpillars network.

    lgraph = layerGraph();
    
    tempLayers = [
        imageInputLayer([P 2],"Name","pillars|indices|reshape","Normalization","none")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        imageInputLayer([P N 9],"Name","pillars|input","Normalization","none")
        convolution2dLayer([1 1],64,"Name","pillars|conv2d","BiasLearnRateFactor",0)
        batchNormalizationLayer("Name","pillars|batchnorm","Epsilon",0.001)
        reluLayer("Name","pillars|relu")
        maxPooling2dLayer([1 N],"Name","pillars|reshape","Stride",[1 N])];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        helperScatterLayer(2,"pillars|scatter_nd",[gridParams{1,4}{1} gridParams{1,4}{2}])
        convolution2dLayer([3 3],64,"Name","cnn|block1|conv2d0","Padding","same","Stride",[2 2])
        reluLayer("Name","cnn|block1|conv2d0_relu")
        batchNormalizationLayer("Name","cnn|block1|bn0","Epsilon",0.001)
        convolution2dLayer([3 3],64,"Name","cnn|block1|conv2d1","Padding","same")
        reluLayer("Name","cnn|block1|conv2d1_relu")
        batchNormalizationLayer("Name","cnn|block1|bn1","Epsilon",0.001)
        convolution2dLayer([3 3],64,"Name","cnn|block1|conv2d2","Padding","same")
        reluLayer("Name","cnn|block1|conv2d2_relu")
        batchNormalizationLayer("Name","cnn|block1|bn2","Epsilon",0.001)
        convolution2dLayer([3 3],64,"Name","cnn|block1|conv2d3","Padding","same")
        reluLayer("Name","cnn|block1|conv2d3_relu")
        batchNormalizationLayer("Name","cnn|block1|bn3","Epsilon",0.001)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        transposedConv2dLayer([3 3],128,"Name","cnn|up1|conv2dt","Cropping","same")
        reluLayer("Name","cnn|up1|conv2dt_relu")
        batchNormalizationLayer("Name","cnn|up1|bn","Epsilon",0.001)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([3 3],128,"Name","cnn|block2|conv2d0","Padding","same","Stride",[2 2])
        reluLayer("Name","cnn|block2|conv2d0_relu")
        batchNormalizationLayer("Name","cnn|block2|bn0","Epsilon",0.001)
        convolution2dLayer([3 3],128,"Name","cnn|block2|conv2d1","Padding","same")
        reluLayer("Name","cnn|block2|conv2d1_relu")
        batchNormalizationLayer("Name","cnn|block2|bn1","Epsilon",0.001)
        convolution2dLayer([3 3],128,"Name","cnn|block2|conv2d2","Padding","same")
        reluLayer("Name","cnn|block2|conv2d2_relu")
        batchNormalizationLayer("Name","cnn|block2|bn2","Epsilon",0.001)
        convolution2dLayer([3 3],128,"Name","cnn|block2|conv2d3","Padding","same")
        reluLayer("Name","cnn|block2|conv2d3_relu")
        batchNormalizationLayer("Name","cnn|block2|bn3","Epsilon",0.001)
        convolution2dLayer([3 3],128,"Name","cnn|block2|conv2d4","Padding","same")
        reluLayer("Name","cnn|block2|conv2d4_relu")
        batchNormalizationLayer("Name","cnn|block2|bn4","Epsilon",0.001)
        convolution2dLayer([3 3],128,"Name","cnn|block2|conv2d5","Padding","same")
        reluLayer("Name","cnn|block2|conv2d5_relu")
        batchNormalizationLayer("Name","cnn|block2|bn5","Epsilon",0.001)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        transposedConv2dLayer([3 3],128,"Name","cnn|up2|conv2dt","Cropping","same","Stride",[2 2])
        reluLayer("Name","cnn|up2|conv2dt_relu")
        batchNormalizationLayer("Name","cnn|up2|bn","Epsilon",0.001)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([3 3],128,"Name","cnn|block3|conv2d0","Padding","same","Stride",[2 2])
        reluLayer("Name","cnn|block3|conv2d0_relu")
        batchNormalizationLayer("Name","cnn|block3|bn0","Epsilon",0.001)
        convolution2dLayer([3 3],128,"Name","cnn|block3|conv2d1","Padding","same")
        reluLayer("Name","cnn|block3|conv2d1_relu")
        batchNormalizationLayer("Name","cnn|block3|bn1","Epsilon",0.001)
        convolution2dLayer([3 3],128,"Name","cnn|block3|conv2d2","Padding","same")
        reluLayer("Name","cnn|block3|conv2d2_relu")
        batchNormalizationLayer("Name","cnn|block3|bn2","Epsilon",0.001)
        convolution2dLayer([3 3],128,"Name","cnn|block3|conv2d3","Padding","same")
        reluLayer("Name","cnn|block3|conv2d3_relu")
        batchNormalizationLayer("Name","cnn|block3|bn3","Epsilon",0.001)
        convolution2dLayer([3 3],128,"Name","cnn|block3|conv2d4","Padding","same")
        reluLayer("Name","cnn|block3|conv2d4_relu")
        batchNormalizationLayer("Name","cnn|block3|bn4","Epsilon",0.001)
        convolution2dLayer([3 3],128,"Name","cnn|block3|conv2d5","Padding","same")
        reluLayer("Name","cnn|block3|conv2d5_relu")
        batchNormalizationLayer("Name","cnn|block3|bn5","Epsilon",0.001)
        transposedConv2dLayer([3 3],128,"Name","cnn|up3|conv2dt","Cropping","same","Stride",[4 4])
        reluLayer("Name","cnn|up3|conv2dt_relu")
        batchNormalizationLayer("Name","cnn|up3|bn","Epsilon",0.001)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = depthConcatenationLayer(3,"Name","cnn|concatenate");
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],3*numAnchors,"Name","size|conv2d",'Weights',randn([1 1 384 3*numAnchors])*0.001)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],3*numAnchors,"Name","loc|conv2d",'Weights',randn([1 1 384 3*numAnchors])*0.001)];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],numAnchors,"Name","occupancy|conv2d")
        sigmoidLayer('Name',"occupancy|conv2dSigmoid")];
    lgraph = addLayers(lgraph,tempLayers);
    
    tempLayers = [
        convolution2dLayer([1 1],numClasses*numAnchors,"Name","clf|conv2d")
        sigmoidLayer('Name',"activation")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = [
        convolution2dLayer([1 1],numAnchors,"Name","heading|conv2d")
        sigmoidLayer('Name',"heading|conv2dSigmoid")];
    lgraph = addLayers(lgraph,tempLayers);

    tempLayers = convolution2dLayer([1 1],numAnchors,"Name","angle|conv2d");                 
    lgraph = addLayers(lgraph,tempLayers);

    % clean up helper variable.
    clear tempLayers;
    
    % Make the necessary layer connections.
    lgraph = connectLayers(lgraph,"pillars|indices|reshape","pillars|scatter_nd/in2");
    lgraph = connectLayers(lgraph,"pillars|reshape","pillars|scatter_nd/in1");
    lgraph = connectLayers(lgraph,"cnn|block1|bn3","cnn|up1|conv2dt");
    lgraph = connectLayers(lgraph,"cnn|block1|bn3","cnn|block2|conv2d0");
    lgraph = connectLayers(lgraph,"cnn|up1|bn","cnn|concatenate/in1");
    lgraph = connectLayers(lgraph,"cnn|block2|bn5","cnn|up2|conv2dt");
    lgraph = connectLayers(lgraph,"cnn|block2|bn5","cnn|block3|conv2d0");
    lgraph = connectLayers(lgraph,"cnn|up2|bn","cnn|concatenate/in2");
    lgraph = connectLayers(lgraph,"cnn|up3|bn","cnn|concatenate/in3");
    lgraph = connectLayers(lgraph,"cnn|concatenate","size|conv2d");
    lgraph = connectLayers(lgraph,"cnn|concatenate","loc|conv2d");
    lgraph = connectLayers(lgraph,"cnn|concatenate","occupancy|conv2d");
    lgraph = connectLayers(lgraph,"cnn|concatenate","clf|conv2d");
    lgraph = connectLayers(lgraph,"cnn|concatenate","heading|conv2d");
    lgraph = connectLayers(lgraph,"cnn|concatenate","angle|conv2d");

end