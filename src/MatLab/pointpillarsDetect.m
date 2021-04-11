function [bboxes,scores,labels] = pointpillarsDetect(matFile, pillarFeatures, pillarIndices, gridParams, numOutputs, confidenceThreshold, overlapThreshold, classNames)
% The pointpillarsDetect function detects the bounding boxes, scores, and
% labels in the point cloud.

coder.extrinsic('helpergeneratePointPillarDetections');

% Define Anchor Boxes
anchorBoxes = {{1.92, 4.5, 1.69, -1.78, 0}, {1.92, 4.5, 1.69, -1.78, pi/2}, {2.1575, 6.0081, 2.3811, -1.78, 0}, {2.1575, 6.0081, 2.3811, -1.78, pi/2}};

% Predict the output
predictions = pointpillarPredict(matFile,pillarFeatures,pillarIndices,numOutputs);

% Generate Detections
[bboxes,scores,labels] = helpergeneratePointPillarDetections(predictions,gridParams,pillarIndices,...
                         anchorBoxes,confidenceThreshold,overlapThreshold,classNames);

end

function YPredCell = pointpillarPredict(matFile,pillarFeatures,pillarIndices,numOutputs)
% Predict the output of network and extract the following confidence,
% x, y, z, l, w, h, yaw and class.

% load the deep learning network for prediction
persistent net;

if isempty(net)
    net = coder.loadDeepLearningNetwork(matFile);
end

YPredCell = cell(1,numOutputs);
[YPredCell{:}] = predict(net,pillarIndices,pillarFeatures);
end