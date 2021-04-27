%% Loss functions
% Loss functions used in the point pillars network.

function [angLoss, occLoss, locLoss, szLoss, hdLoss, clfLoss]...
    = computePointPillarLoss(YPredictions, YTargets, gridParams, numClasses, numAnchors)
    
    % Create a mask of anchors for which the loss has to computed.
    posMask = YTargets{1,3} == 1;
    
    % Compute the loss.
    occLoss = occupancyLoss(YTargets{1,3}, YPredictions{1,3}, 1.0);
    locLoss = locationLoss(YTargets{1,2}, YPredictions{1,2}, posMask, 2.0);
    szLoss = sizeLoss(YTargets{1,1}, YPredictions{1,1}, posMask,2.0);
    angLoss = angleLoss(YTargets{1,6}, YPredictions{1,6}, posMask,2.0);
    hdLoss = headingLoss(YTargets{1,5}, YPredictions{1,5}, posMask,0.2); 
    clfLoss = classificationLoss(YTargets{1,4}, YPredictions{1,4}, posMask, 1.0, gridParams, numClasses, numAnchors);
end

% Compute the occupancy loss.
function loss = occupancyLoss(Targets, Predictions, focalWeight)
    loss = focalCrossEntropy(Predictions,Targets,'TargetCategories', ...
        'independent','Reduction','none');
    posMask = (Targets == 1) | (Targets == 0);
    loss = loss .* posMask;
    nanInd = isnan(extractdata(loss));
    loss(nanInd) = 0;
    dFactor = max(sum(posMask,'all'),1);
    loss = sum(loss,'all')/dFactor;
    loss = loss * focalWeight;
end

% Compute the location loss.
 function loss = locationLoss(Targets, Predictions, mask, locWeight)
    mask = repmat(mask,[1,1,3,1]);
    loss = huber(Predictions, Targets, 'Reduction', 'none');
    loss = loss .* mask;
    nanInd = isnan(extractdata(loss));
    loss(nanInd) = 0;
    dFactor = sum(mask,'all');
    loss = sum(loss,'all')/dFactor;
    
    loss = loss * locWeight;
 end

% Compute the size loss.
function loss = sizeLoss(Targets, Predictions, mask, sizeWeight)
    mask = repmat(mask,[1,1,3,1]);
    loss = huber(Predictions, Targets, 'Reduction', 'none');
    loss = loss .* mask;
    nanInd = isnan(extractdata(loss));
    loss(nanInd) = 0;
    dFactor = sum(mask,'all');
    loss = sum(loss,'all')/dFactor;
    
    loss = loss * sizeWeight;
end

% Compute the angle loss.
function loss = angleLoss(Targets, Predictions, mask, angleWeight)
    loss = huber(Predictions, Targets, 'Reduction', 'none');
    loss = loss .* mask;
    nanInd = isnan(extractdata(loss));
    loss(nanInd) = 0;
    dFactor = sum(mask,'all');
    loss = sum(loss,'all')/dFactor;
    
    loss = loss * angleWeight;
end

% Compute the heading loss.
function loss = headingLoss(Targets, Predictions, mask, headingWeight)
    loss = focalCrossEntropy(Predictions,Targets,'Gamma',0,...
           'Alpha',1,'TargetCategories','independent','Reduction','none');
    loss = loss .* mask;
    nanInd = isnan(extractdata(loss));
    loss(nanInd) = 0;
    dFactor = sum(mask,'all');
    loss = sum(loss,'all')/dFactor;
    loss = loss * headingWeight;
end

function res = reshapeMatForClfLoss(mat, batchSize,numClasses, numAnchors, gridX, gridY)
    temp = reshape(mat, gridX, gridY, numAnchors, numClasses, batchSize);
    if(batchSize == 1)
        temp = permute(temp, [3,1,2,4]);
    else
        temp = permute(temp, [3,1,2,4,5]);
    end
    res = reshape(temp, [], numClasses, batchSize);
end

% Compute the classification loss
function loss = classificationLoss(Targets, Predictions, mask, ...
    classWeight, gridParams, numClasses, numAnchors)

    if(length(size(Targets)) > 3)
        batchSize = size(Targets, 4);
    else
        batchSize = 1;
    end

    gridX = gridParams{1,4}{1}/gridParams{1,3}{3};
    gridY = gridParams{1,4}{2}/gridParams{1,3}{3};
    
    % Reshape Targets and Predictions to requires format for crossentropy
    nTarg = reshapeMatForClfLoss(Targets, batchSize, numClasses, numAnchors, gridX, gridY);
    nPred = reshapeMatForClfLoss(Predictions, batchSize, numClasses, numAnchors, gridX, gridY);

    loss = crossentropy(nPred, nTarg, 'DataFormat', 'SCB', ...
        'TargetCategories', 'independent', 'Reduction', 'none');
    
    % Reshape mask to the required format
    mask = repmat(mask,1,1,numClasses,1);
    mask = reshapeMatForClfLoss(mask, batchSize, numClasses, numAnchors, gridX, gridY);  
    
    loss = loss .* mask;
    nanInd = isnan(extractdata(loss));
    loss(nanInd) = 0;
    dFactor = sum(mask,'all');
    loss = sum(loss,'all')/dFactor;
    loss = loss * classWeight;
end
