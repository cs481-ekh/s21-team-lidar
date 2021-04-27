%% Create Pillars and Pillar Indices from the point cloud
% This function takes in the point cloud and grid parameters as the input
% to compute the pillar features and pillar indices.

function dataAug = createPillars(data,gridParams,P,N)   
% Create Pillars and Pillar Indices from the point cloud.
    
    numObservations = size(data,1);    
    spatialLimits = [gridParams{1,1}{1}, gridParams{1,2}{1};
                     gridParams{1,1}{2}, gridParams{1,2}{2};
                     gridParams{1,1}{3}, gridParams{1,2}{3}];        
    gridSize = [gridParams{1,4}{1} gridParams{1,4}{2} 1];
    
    % Adding two more dimension in the cell for adding pillars and pillar
    % indices.
    dataAug = cell(size(data,1), size(data,2)+1);
    
    for obs = 1:numObservations
        pc = data{obs,1};
        ptCld = pointCloud(pc(:,1:3));
        ptCld.Intensity = pc(:,4);
        
        [grid, gridLocations] = pcbin(ptCld, gridSize, spatialLimits);
        pointPillars = cell(size(grid));
        
        % Calculate the pillar features from each pillar.
        for i = 1:numel(grid)
            pointPillars{i} = buildPointPillarFeatures(grid{i},gridLocations{i},ptCld,N);
        end
        
        % Find the grid indices that are not empty.
        nonEmptyPillarMask = ~cellfun(@isempty,pointPillars);
        nonemptyPillarIndices = find(nonEmptyPillarMask);
        nonemptyPillarIndices = nonemptyPillarIndices(randperm(min(length(nonemptyPillarIndices),P)));
        sampledPillars = pointPillars(nonemptyPillarIndices);
        
        % Sample the points from non empty pillars.
        op = cellfun(@(c) samplePointsFromPillars(c,N) ,sampledPillars,'UniformOutput',false);
        pillarFeatures = cat(3,op{:}); 
        pillarFeatures = permute(pillarFeatures,[3,1,2]);
        pillarEncodedFeatures = zeros(P,N,9);
        numPillars = size(pillarFeatures,1);
        if(~isempty(pillarFeatures))
            pillarEncodedFeatures(1:numPillars,:,:) = pillarFeatures;
        end
         
        numPillars = size(nonemptyPillarIndices,1);
        
        % Find the indices of the non empty pillars.
        [pillarRows,pillarCols] = ind2sub([gridSize(1,1) gridSize(1,2)],nonemptyPillarIndices);
        pillarConcat = cat(2,pillarRows,pillarCols);
        pillarIndices = zeros(P,2);
        
        if(~isempty(pillarIndices))
            pillarIndices(1:numPillars,:) = pillarConcat;
        end
        
        dataAug{obs,1} = pillarEncodedFeatures;
        dataAug{obs,2} = squeeze(pillarIndices);
        dataAug{obs,3} = data{obs,2};
        dataAug{obs,4} = data{obs,3};
    end

end

function output = samplePointsFromPillars(featurePoints,N)
    if size(featurePoints,1) >= N
        idx = randperm(size(featurePoints,1));
        output = featurePoints(idx(1:N),:);
    else
        % Apply zero padding to yield N points from pillar.
        amountToPad = N-size(featurePoints,1);
        output = cat(1,featurePoints,zeros(amountToPad,size(featurePoints,2)));
    end
end


function outputFeatures = buildPointPillarFeatures(idx,binLocation,pc,N)
    outputFeatures = [];
    if isempty(idx)
        return
    end
    
    % Compute the features from the points in the pillars.
    points = pc.Location(idx,:);
    pointIntensity = pc.Intensity(idx);
    centroid = sum(points,1)/N;
    distanceToPointsCentroid = points-centroid;
    offsetFromPillarCenter = points(:,1:2)-((binLocation(1:2,1) + binLocation(1:2,2))/2)';
    outputFeatures = cat(2,points,pointIntensity,distanceToPointsCentroid,offsetFromPillarCenter);
end
