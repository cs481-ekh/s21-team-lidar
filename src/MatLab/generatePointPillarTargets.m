function boxTargets = generatePointPillarTargets(YPredictions, boxLabels, pillarIndices, gridParams, anchorBoxes, numClasses)
% Creates targets for every prediction element x, y, z, length, width,
% height, yawangle, confidence and class probabilities.

    sizeTarget = zeros(size(YPredictions{1,1}));
    locTarget = zeros(size(YPredictions{1,2}));
    occTarget = zeros(size(YPredictions{1,3}));
    clfTarget = zeros(size(YPredictions{1,4}));
    hdTarget = zeros(size(YPredictions{1,5}));
    angTarget = zeros(size(YPredictions{1,6}));
    
    batchSize = size(pillarIndices,4);
    for i = 1:batchSize
        idx = (boxLabels(:,10,1,i) ~= 0);
        gtBoxes = boxLabels(idx,1:9,1,i);
        gtClasses = boxLabels(idx,10,1,i);
        targets = generateTargets(gtBoxes,pillarIndices(:,:,:,i),gridParams,anchorBoxes,gtClasses,numClasses);
        
        gridX = size(targets,1);
        gridY = size(targets,2);
        sizeTarget(:,:,:,i) = reshape(targets(:,:,:,5:7),gridX,gridY,[]);
        locTarget(:,:,:,i) = reshape(targets(:,:,:,2:4),gridX,gridY,[]);
        occTarget(:,:,:,i) = reshape(targets(:,:,:,1),gridX,gridY,[]);
        clfTarget(:,:,:,i) = reshape(targets(:,:,:,10:end),gridX,gridY,[]);
        hdTarget(:,:,:,i) = reshape(targets(:,:,:,9),gridX,gridY,[]);
        angTarget(:,:,:,i) = reshape(targets(:,:,:,8),gridX,gridY,[]);
    end
    
    boxTargets = {sizeTarget,locTarget,occTarget,clfTarget,hdTarget,angTarget};
end

function boxTargets = generateTargets(gtBoxes,pillarIndices,gridParams,anchorBoxes,gtClasses,numClasses)
    
    % Extract the data from gpuArray and dlArray.
    gtBoxes = gather(extractdata(gtBoxes));
    gtClasses = gather(extractdata(gtClasses));   
    pillarIndices = gather(extractdata(pillarIndices));
    
    % Convert the yaw angle from degrees to radians.
    gtBoxes(:,9) = deg2rad(gtBoxes(:,9));
    gtBoxes(:,9) = wrapToPi(gtBoxes(:,9));
    
    % Generate the tiled anchors from grid parameters and anchor box
    % parmaeters.
    [anchors3D, anchorsBEV] = createAnchors(gridParams, anchorBoxes);
    numAnchors = size(anchors3D,1);
    
    % Generate the anchor mask so that overlap with ground truth boxes is
    % calculated on the positive anchors.
    anchorMask = createAnchorMask(pillarIndices, gridParams, anchorsBEV);
    anchorMask = anchorMask > 1;
    numInside = sum(anchorMask,'all');
    insideAnchors = anchors3D(anchorMask,:);
    labels = -1*ones(numInside,1);
    
    % Extract the bird's eye coordinates from the anchors and groundtruth
    % boxes.
    insideAnchorsBEV = insideAnchors(:,[1,2,4,5,7]);
    gtBoxesBEV = gtBoxes(:,[1,2,4,5,9]);
    
    % Convert anchor boxes to nearest 2-D bounding boxes.
    rots = insideAnchorsBEV(:,5);
    rots = rots - floor((rots+0.5)/pi)*pi;
    idx = (rots > pi/4);
    insideAnchorsBEV(idx,:) = insideAnchorsBEV(idx,[1,2,4,3,5]);
    
    % Convert groundtruth boxes to nearest 2-D bounding boxes.
    rots = gtBoxesBEV(:,5);
    rots = rots - floor((rots+0.5)/pi)*pi;
    idx = (rots > pi/4);
    gtBoxesBEV(idx,:) = gtBoxesBEV(idx,[1,2,4,3,5]);
    
    % Calculate the overlap ratio between anchors and ground truth boxes.
    overlapAnchorGt = bboxOverlapRatio(insideAnchorsBEV(:,[1,2,3,4]),gtBoxesBEV(:,[1,2,3,4]));
    
    % Get the iou of best matching anchor box.
    [anchorGtMax,anchorGtArgmax] = max(overlapAnchorGt,[],2);
    [~,gtAnchorArgmax] = max(overlapAnchorGt,[],1);
    [anchorswithMaxOverlap,~] = ind2sub([numInside,size(gtBoxes,2)],gtAnchorArgmax);
    posInds = anchorGtMax > 0.55;
    
    % Set labels to 0 when overlap ratio is less than 0.3.
    labels(anchorGtMax < 0.3) = 0;
    labels(posInds) = 1;
    
    % Set labels to 1 when overlap ratio is more than 0.6.
    labels(anchorswithMaxOverlap) = 1; 
    
    % Remove labels as positive which are not overlapping with any of the
    % anchors.
    labels(anchorGtMax == 0) = 0;           
    fgInds = find(labels>0);
        
    boxTargets = zeros(numInside,9+numClasses);
    boxEncodings = createBoxEncodings(gtBoxes(anchorGtArgmax(fgInds),:),insideAnchors(fgInds,:),gtClasses(anchorGtArgmax(fgInds),:),numClasses);
    boxTargets(fgInds,:) = boxEncodings;
    boxTargets = unmap(boxTargets, numAnchors, anchorMask, 0);
    labels = unmap(labels, numAnchors, anchorMask, -1);
    boxTargets(:,1) = labels;
    
    gridX = gridParams{1,4}{1}/gridParams{1,3}{3};
    gridY = gridParams{1,4}{2}/gridParams{1,3}{3};
    boxTargets = reshape(boxTargets,[numel(anchorBoxes),gridX,gridY,9+numClasses]);
    boxTargets = permute(boxTargets,[2,3,1,4]);
    
end

function outputData = unmap(data, count, inds, fill)
    if count == sum(inds)
        outputData = data;
        return;
    end
    sz = [count,size(data,2)];
    outputData = fill*ones(sz);
    outputData(inds,:) = data;
end

%% * Generate Box Encodings from labels and targets *
% Generate the encodings from the targets and the corresponding anchor
% boxes and the ground truth classes.

function endcodings = createBoxEncodings(labelBoxes,anchorBoxes,gtClasses,numClasses)
    % Convert categorical labels to one-hot vectors.
    gtClasses = onehotencode(categorical(gtClasses), 2);
    
    endcodings = zeros(size(labelBoxes,1),9+numClasses);
    
    for i = 1:size(labelBoxes,1)
        endcodings(i,1) = 1;
        diag = sqrt(anchorBoxes(i,4)^2 + anchorBoxes(i,5)^2);
        
        % Encode the location of the boxes.
        endcodings(i,2) = (labelBoxes(i,1) - anchorBoxes(i,1))/diag;
        endcodings(i,3) = (labelBoxes(i,2) - anchorBoxes(i,2))/diag;
        endcodings(i,4) = (labelBoxes(i,3) - anchorBoxes(i,3))/anchorBoxes(i,6);
        
        % Encode the dimensions of the boxes.
        endcodings(i,5) = log(labelBoxes(i,4)/anchorBoxes(i,4));
        endcodings(i,6) = log(labelBoxes(i,5)/anchorBoxes(i,5));
        endcodings(i,7) = log(labelBoxes(i,6)/anchorBoxes(i,6));
        
        % Encode the yaw angle for the boxes.
        endcodings(i,8) = sin(labelBoxes(i,9) - anchorBoxes(i,7));
        
        % Encode the heading value for the box based on the yaw angle.
        if labelBoxes(i,9) > 0
            endcodings(i,9) = 1;
        else
            endcodings(i,9) = 0;
        end
        
        % Assign the ground truth class to the box.
        endcodings(i,10:end) = gtClasses(i,:);        
    end
end
