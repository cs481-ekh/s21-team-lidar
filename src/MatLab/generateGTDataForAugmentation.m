function groundTruthData = generateGTDataForAugmentation(processedData, groundTruth)
% This function is used to extract all the ground truth labels from the
% provided input data.

    groupDbInfos = struct;
    numFiles = size(processedData,1);

    classNames = groundTruth.Properties.VariableNames;
    for j = 1:numel(classNames)
        Data = {};
        for i = 1:numFiles     
            finalPC = processedData{i,1};   
            labels = groundTruth(i,j).Variables;
            labels = labels{1};
               for ii = 1:size(labels,1)
                   label = labels(ii,:);
                   
                   % Find points that are present inside the cuboid and set
                   % the difficulty level.
                   pointsIdx = findPointsInsideCuboid(cuboidModel(label),finalPC);
                   data = struct;
                   data.numPointsInGt = size(pointsIdx,1);
                   loc = cat(2,finalPC.Location,finalPC.Intensity);
                   data.lidarpoints = loc(pointsIdx,:);
                   if size(pointsIdx,1) < 20
                       data.difficulty = -1;
                   else
                       data.difficulty = 1;
                   end
                   data.boxDims = labels(ii,[1,2,3,4,5,6,9]);
                   data.className = classNames{j};
                   Data{end+1} = data;
                end 
        end
        groupDbInfos.(classNames{j}) =  Data;
    end
    fnames = fieldnames(groupDbInfos);
    groundTruthData = struct;
    for i = 1:numel(fnames)
        fname = fnames{i};
        sampledList = groupDbInfos.(fname);
        groundTruthData.(fname) = batchSampler(sampledList, fname);
    end
end