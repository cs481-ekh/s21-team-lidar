%% Add ground truth labels to the current point cloud
% Functions samples specified number of ground truth labels from the
% database and performs collision test and returns the samples that are
% non-overlapping.
function data = groundTruthDataAugmenation(data,groundTruthData,samplesToAdd)
    ptCloud = data{1,1};
    
    % Convert to Mx4 format from point cloud object.
    ptCloud = cat(2,ptCloud.Location,ptCloud.Intensity);
    bboxesGt = data{1,2};
    bboxesClasses = data{1,3};
    
    fieldNames = fieldnames(samplesToAdd);

    for i = 1:numel(fieldNames)            

        % To remove the pitch and roll from the box coordinates.
        if ~isempty(bboxesGt)
            bboxesGt = bboxesGt(:,[1,2,3,4,5,6,9]);
        end

        % Calculate the number of boxes to add.
        numPCToAdd = samplesToAdd.(fieldNames{i}) - sum(data{1,3} == fieldNames{i});
        numPCToAdd = max(numPCToAdd,0); 
        bboxesToAdd = [];
        pcToAdd = [];

        % Sample the required number of ground truth objects.
        samples = groundTruthData.(fieldNames{i}).sample(numPCToAdd);

        for ii = 1:numel(samples)
            difficulty = samples(ii,1).difficulty; 
            if difficulty == -1 && ~isempty(bboxesGt)
                continue;
            end
            bboxesToAdd = [bboxesToAdd;samples(ii,1).boxDims(:,[1,2,3,4,5,6,7])];
            pcToAdd = [pcToAdd;{samples(ii,1).lidarpoints}];
        end

        % Find overlap within the selected boxes.
        if ~isempty(bboxesToAdd)
            bboxesBEV = bboxesToAdd(:,[1,2,4,5,7]);
            boxscores  = rand(size(bboxesBEV,1),1);
            [~,~,idx] = selectStrongestBbox(bboxesBEV,boxscores,'OverlapThreshold',0);
            bboxesToAdd = bboxesToAdd(idx,:);
            pcToAdd = pcToAdd(idx);
        end

       % Condition wherein both ground truth boxes and boxes to add
       % are present.
       if ~isempty(bboxesGt) && ~isempty(bboxesToAdd)
            overlapRatio = bboxOverlapRatio(bboxesToAdd(:,[1,2,4,5,7]),bboxesGt(:,[1,2,4,5,7]));
            maxOverlap = max(overlapRatio,[],2);
            bboxesToAddFinal = bboxesToAdd(maxOverlap == 0,:);
            pcToAddFinal = pcToAdd(maxOverlap == 0);
            finalGtBoxes = [bboxesGt;bboxesToAddFinal];
            pcToAddClasses = cell(size(bboxesToAddFinal,1),1);
            pcToAddClasses(:) = {fieldNames{i}};
            pcToAddClasses = categorical(pcToAddClasses);
            finalClasses = cat(1,bboxesClasses,pcToAddClasses);

        % Condition wherein there are no ground truth boxes.
        elseif isempty(bboxesGt)
            bboxesToAddFinal = bboxesToAdd;
            pcToAddFinal = pcToAdd;
            finalGtBoxes = bboxesToAdd;
            pcToAddClasses = cell(size(finalGtBoxes,1),1);
            pcToAddClasses(:) = {fieldNames{i}};
            pcToAddClasses = categorical(pcToAddClasses);
            finalClasses = pcToAddClasses;

        % Condition wherein there are no bboxes to add to the
        % existing ground truths.
        elseif isempty(bboxesToAdd)
            finalGtBoxes = bboxesGt;
            bboxesToAddFinal = [];
            finalClasses = bboxesClasses;
        end

        % Extract the points inside the boxes and append to point
        % cloud.
        if ~isempty(bboxesToAddFinal)
           for ii = 1:size(bboxesToAddFinal,1)
                pointsToAdd = pcToAddFinal(ii);
                pointsToAdd = pointsToAdd{1,1};
                if isempty(pointsToAdd)
                    continue;
                end
                ptCloud = [ptCloud;pointsToAdd];
            end
        end

        finalBoxes = zeros([size(finalGtBoxes,1),9]);
        finalBoxes(:,1:6) = finalGtBoxes(:,1:6);
        finalBoxes(:,9) = finalGtBoxes(:,7);
        bboxesGt = finalBoxes;
        bboxesClasses = finalClasses;
    end

    data{1,1} = ptCloud;
    if isempty(finalBoxes)
        fprintf("Error in finalBoxes Augmentation");
    elseif size(finalBoxes,1) ~= size(finalClasses,1)
        fprintf("Error in final Classes Augmentation");
    end
    data{1,2} = finalBoxes;
    data{1,3} = finalClasses;
end