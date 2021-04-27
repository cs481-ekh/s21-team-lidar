function [croppedPointCloudObj, processedLabels] = cropFrontViewFromLidarData(lidarData, boxLabels, gridParams)
% This function crops the front view from the input full-view point cloud
% and also processes the corresponding box labels according to the 
% specified grid parameters.

    tmpStr = '';
    numFiles = size(boxLabels,1);
    
    processedLabels = cell(size(boxLabels));
    croppedPointCloudObj = cell(size(numFiles));

    % Get the classes from the ground truth labels.
    classNames = boxLabels.Properties.VariableNames;
    
    for i = 1:numFiles

        ptCloud = read(lidarData);            
        groundTruth = boxLabels(i,:);
        
        % Set the limits for the point cloud.
        [x,y] = find( ptCloud.Location(:,:,1) < gridParams{1,2}{1} ...
                            & ptCloud.Location(:,:,1) > gridParams{1,1}{1} ...
                            & ptCloud.Location(:,:,2) < gridParams{1,2}{2} ...
                            & ptCloud.Location(:,:,2) > gridParams{1,1}{2} ...
                            & ptCloud.Location(:,:,3) < gridParams{1,2}{3} ...
                            & ptCloud.Location(:,:,3) > gridParams{1,1}{3});    
        ptCloud = select(ptCloud, x, y, 'OutputSize', 'full'); 
        processedData = removeInvalidPoints(ptCloud);
         
        for ii = 1:numel(classNames)

            labels = groundTruth(1,classNames{ii}).Variables;
            if(iscell(labels))
                labels = labels{1};
            end
            if ~isempty(labels)

                % Get the label indices that are in the selected RoI.
                labelsIndices = labels(:,1) > gridParams{1,1}{1} ...
                            & labels(:,1) < gridParams{1,2}{1} ...
                            & labels(:,2) > gridParams{1,1}{2} ...
                            & labels(:,2) < gridParams{1,2}{2} ...
                            & labels(:,4) > 0 ...
                            & labels(:,5) > 0 ...
                            & labels(:,6) > 0;
                labels = labels(labelsIndices,:);

                if ~isempty(labels)
                    % Find the number of points inside each ground truth
                    % label.
                    numPoints = arrayfun(@(x)(findPointsInsideCuboid(cuboidModel(labels(x,:)),processedData)),...
                                (1:size(labels,1)).','UniformOutput',false);

                    posLabels = cellfun(@(x)(length(x) > 50), numPoints);
                    labels = labels(posLabels,:);
                end
            end
            processedLabels{i,ii} = labels;
        end
        croppedPointCloudObj{i,1} = processedData;
    end
    
    % Print completion message when done.
    msg = sprintf('Processing data 100%% complete');
    fprintf(1,'%s',[tmpStr, msg]);

    processedLabels = cell2table(processedLabels);
    numClasses = size(processedLabels,2);
    for j = 1:numClasses
        processedLabels.Properties.VariableNames{j} = classNames{j};
    end

end