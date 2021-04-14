function success = pp() 
    f = waitbar(0,'Loading Point Cloud Data...');
    mkdir("C:\Users\andres\Documents\PointPillars");
    outputFolder = fullfile("C:\Users\andres\Documents\PointPillars");
    pretrainedNetURL = 'https://ssd.mathworks.com/supportfiles/lidar/data/trainedPointPillars.zip';

    doTraining = false;
    if ~doTraining
        net = downloadPretrainedPointPillarsNet(outputFolder, pretrainedNetURL);
    end
    
    lidarURL = 'https://www.mathworks.com/supportfiles/lidar/data/WPI_LidarData.tar.gz';
    
    lidarData  = downloadWPIData(outputFolder, lidarURL);

    data = l
    Labels = timetable2table(bboxGroundTruth);
    Labels = Labels(:,2:end);

    figure
    ax = pcshow(lidarData{1,1});
    set(ax,'XLim',[-50 50],'YLim',[-40 40]);
    zoom(ax,2.5);
    axis off;

    xMin = 0.0;     % Minimum value along X-axis.
    yMin = -39.68;  % Minimum value along Y-axis.
    zMin = -5.0;    % Minimum value along Z-axis.
    xMax = 69.12;   % Maximum value along X-axis.
    yMax = 39.68;   % Maximum value along Y-axis.
    zMax = 5.0;     % Maximum value along Z-axis.
    xStep = 0.16;   % Resolution along X-axis.
    yStep = 0.16;   % Resolution along Y-axis.
    dsFactor = 2.0; % Downsampling factor.

    % Calculate the dimensions for pseudo-image.
    Xn = round(((xMax - xMin) / xStep));
    Yn = round(((yMax - yMin) / yStep));

    % Define pillar extraction parameters.
    gridParams = {{xMin,yMin,zMin},{xMax,yMax,zMax},{xStep,yStep,dsFactor},{Xn,Yn}};
    waitbar(.1,'Processing Point Cloud Data...');

    % Load the calibration parameters.
    fview = load('calibrationValues.mat');
    [inputPointCloud, boxLabels] = createFrontViewFromLidarData(lidarData, Labels, gridParams, fview); 

    figure
    ax1 = pcshow(inputPointCloud{1,1}.Location);
    gtLabels = boxLabels.car(1,:);
    showShape('cuboid', gtLabels{1,1}, 'Parent', ax1, 'Opacity', 0.1, 'Color', 'green','LineWidth',0.5);
    zoom(ax1,2);

    rng(1);
    shuffledIndices = randperm(size(inputPointCloud,1));
    idx = floor(0.7 * length(shuffledIndices));

    trainData = inputPointCloud(shuffledIndices(1:idx),:);
    testData = inputPointCloud(shuffledIndices(idx+1:end),:);
    
    trainLabels = boxLabels(shuffledIndices(1:idx),:);
    testLabels = boxLabels(shuffledIndices(idx+1:end),:);

    dataLocation = fullfile(outputFolder,'InputData');
    saveptCldToPCD(trainData,dataLocation);

    lds = fileDatastore(dataLocation,'ReadFcn',@(x) pcread(x));

    bds = boxLabelDatastore(trainLabels);

    cds = combine(lds,bds);

    augData = read(cds);
    augptCld = augData{1,1};
    augLabels = augData{1,2};
    figure;
    ax2 = pcshow(augptCld.Location);
    showShape('cuboid', augLabels, 'Parent', ax2, 'Opacity', 0.1, 'Color', 'green','LineWidth',0.5);
    zoom(ax2,2);

    reset(cds);

    gtData = generateGTDataForAugmentation(trainData,trainLabels);
    
    cdsAugmented = transform(cds,@(x) groundTruthDataAugmenation(x,gtData));

    cdsAugmented = transform(cdsAugmented,@(x) augmentData(x));
    augData = read(cdsAugmented);
    augptCld = augData{1,1};
    augLabels = augData{1,2};
    figure;
    ax3 = pcshow(augptCld(:,1:3));
    showShape('cuboid', augLabels, 'Parent', ax3, 'Opacity', 0.1, 'Color', 'green','LineWidth',0.5);
    zoom(ax3,2);

    reset(cdsAugmented);

    % Define number of prominent pillars.
    P = 12000; 

    % Define number of points per pillar.
    N = 100;   
    cdsTransformed = transform(cdsAugmented,@(x) createPillars(x,gridParams,P,N));

    anchorBoxes = {{3.9, 1.6, 1.56, -1.78, 0}, {3.9, 1.6, 1.56, -1.78, pi/2}};
    numAnchors = size(anchorBoxes,2);
    classNames = trainLabels.Properties.VariableNames;

    lgraph = pointpillarNetwork(numAnchors,gridParams,P,N);

    numEpochs = 160;
    miniBatchSize = 2;
    learningRate = 0.0002;
    learnRateDropPeriod = 15;
    learnRateDropFactor = 0.8;
    gradientDecayFactor = 0.9;
    squaredGradientDecayFactor = 0.999;
    trailingAvg = [];
    trailingAvgSq = [];

    executionEnvironment = "auto";
    if canUseParallelPool
        dispatchInBackground = true;
    else
        dispatchInBackground = false;
    end

    mbq = minibatchqueue(cdsTransformed,3,...
                         "MiniBatchSize",miniBatchSize,...
                         "OutputEnvironment",executionEnvironment,...
                         "MiniBatchFcn",@(features,indices,boxes,labels) createBatchData(features,indices,boxes,labels,classNames),...
                         "MiniBatchFormat",["SSCB","SSCB",""],...
                         "DispatchInBackground",dispatchInBackground);

    if doTraining
        % Convert layer graph to dlnetwork.
        net = dlnetwork(lgraph);

        % Initialize plot.
        fig = figure;
        lossPlotter = configureTrainingProgressPlotter(fig);    
        iteration = 0;

        % Custom training loop.
        for epoch = 1:numEpochs

            % Reset datastore.
            reset(mbq);

            while(hasdata(mbq))
                iteration = iteration + 1;

                % Read batch of data.
                [pillarFeatures, pillarIndices, boxLabels] = next(mbq);

                % Evaluate the model gradients and loss using dlfeval and the modelGradients function.
                [gradients, loss, state] = dlfeval(@modelGradients, net, pillarFeatures, pillarIndices, boxLabels,...
                                                    gridParams, anchorBoxes, executionEnvironment);

                % Do not update the network learnable parameters if NaN values
                % are present in gradients or loss values.
                if checkForNaN(gradients,loss)
                    continue;
                end

                % Update the state parameters of dlnetwork.
                net.State = state;

                % Update the network learnable parameters using the Adam
                % optimizer.
                [net.Learnables, trailingAvg, trailingAvgSq] = adamupdate(net.Learnables, gradients, ...
                                                                   trailingAvg, trailingAvgSq, iteration,...
                                                                   learningRate,gradientDecayFactor, squaredGradientDecayFactor);

                % Update training plot with new points.         
                addpoints(lossPlotter, iteration,double(gather(extractdata(loss))));
                title("Training Epoch " + epoch +" of " + numEpochs);
                drawnow;
            end

            % Update the learning rate after every learnRateDropPeriod.
            if mod(epoch,learnRateDropPeriod) == 0
                learningRate = learningRate * learnRateDropFactor;
            end
        end
    end

    numInputs = numel(testData);
    
    % Generate rotated rectangles from the cuboid labels.
    bds = boxLabelDatastore(testLabels);
    groundTruthData = transform(bds,@(x) createRotRect(x));

    % Set the threshold values.
    nmsPositiveIoUThreshold = 0.5;
    confidenceThreshold = 0.25;
    overlapThreshold = 0.1;

    % Set numSamplesToTest to numInputs to evaluate the model on the entire
    % test data set.
    numSamplesToTest = 50;
    detectionResults = table('Size',[numSamplesToTest 3],...
                            'VariableTypes',{'cell','cell','cell'},...
                            'VariableNames',{'Boxes','Scores','Labels'});

    for num = 1:numSamplesToTest
        ptCloud = testData{num,1};

        [box,score,labels] = generatePointPillarDetections(net,ptCloud,anchorBoxes,gridParams,classNames,confidenceThreshold,...
                                                overlapThreshold,P,N,executionEnvironment);

        % Convert the detected boxes to rotated rectangles format.
        if ~isempty(box)
            detectionResults.Boxes{num} = box(:,[1,2,4,5,7]);
        else
            detectionResults.Boxes{num} = box;
        end
        detectionResults.Scores{num} = score;
        detectionResults.Labels{num} = labels;
    end

    metrics = evaluateDetectionAOS(detectionResults,groundTruthData,nmsPositiveIoUThreshold);

    ptCloud = testData{3,1};
    gtLabels = testLabels{3,1}{1};

    % Display the point cloud.
    figure;
    ax4 = pcshow(ptCloud.Location);

    % The generatePointPillarDetections function detects the bounding boxes, scores for a
    % given point cloud.
    confidenceThreshold = 0.5;
    overlapThreshold = 0.1;
    [box,score,labels] = generatePointPillarDetections(net,ptCloud,anchorBoxes,gridParams,classNames,confidenceThreshold,...
                          overlapThreshold,P,N,executionEnvironment);

    % Display the detections on the point cloud.
    showShape('cuboid', box, 'Parent', ax4, 'Opacity', 0.1, 'Color', 'red','LineWidth',0.5);hold on;
    showShape('cuboid', gtLabels, 'Parent', ax4, 'Opacity', 0.1, 'Color', 'green','LineWidth',0.5);
    zoom(ax4,2);
    success = 0;
end 

function [processedData, processedLabels] = createFrontViewFromLidarData(ptCloudData, groundTruth, gridParams, fview)
    numFiles = size(ptCloudData,1);
    processedLabels = cell(size(groundTruth));
    processedData = cell(size(ptCloudData));

    theta = pi;
    rot = [cos(theta) sin(theta) 0; ...
          -sin(theta) cos(theta) 0; ...
                   0  0  1];
    trans = [0, 0, 0];
    tform = rigid3d(rot, trans);
    tmpStr = '';
    for i = 1:numFiles
        ptCloud = ptCloudData{i,1};
        ptCloud = pctransform(ptCloud,tform);

        % Get the indices of the point cloud that constitute the RoI
        % defined in the calibration parameters.
        [~, indices] = projectLidarPointsOnImage(ptCloud,fview.cameraParams, rigid3d(fview.tform.T));
        ptCloudTransformed = select(ptCloud, indices,'outputSize','full');
        ptCloudTransformed = pctransform(ptCloudTransformed, tform);

        % Set the limits for the point cloud.
        [row, column] = find( ptCloudTransformed.Location(:,:,1) < gridParams{1,2}{1} ...
                            & ptCloudTransformed.Location(:,:,1) > gridParams{1,1}{1} ...
                            & ptCloudTransformed.Location(:,:,2) < gridParams{1,2}{2} ...
                            & ptCloudTransformed.Location(:,:,2) > gridParams{1,1}{2} ...
                            & ptCloudTransformed.Location(:,:,3) < gridParams{1,2}{3} ...
                            & ptCloudTransformed.Location(:,:,3) > gridParams{1,1}{3});    
        ptCloudTransformed = select(ptCloudTransformed, row, column, 'OutputSize', 'full'); 
        finalPC = removeInvalidPoints(ptCloudTransformed);

        % Get the classes from the ground truth labels.
        classNames = groundTruth.Properties.VariableNames; 
        for ii = 1:numel(classNames)

            labels = groundTruth(i,classNames{ii}).Variables;
            labels = labels{1};
            if ~isempty(labels)

                % Get the label indices that are in the selected RoI.
                labelsIndices = labels(:,1) > gridParams{1,1}{1} ...
                            & labels(:,1) < gridParams{1,2}{1} ...
                            & labels(:,2) > gridParams{1,1}{2} ...
                            & labels(:,2) < gridParams{1,2}{2};
                labels = labels(labelsIndices,:);

                % Change the dimension of the ground truth object to fixed
                % value.
                carNewDim = repmat([3.9 1.6 1.56], size(labels,1),1);
                labels(:,4:6) = carNewDim;

                if ~isempty(labels)
                    % Find the number of points inside each ground truth
                    % label.
                    numPoints = arrayfun(@(x)(findPointsInsideCuboid(cuboidModel(labels(x,:)),finalPC)),...
                                (1:size(labels,1)).','UniformOutput',false);
                    posLabels = cellfun(@(x)(~isempty(x)),numPoints);
                    labels = labels(posLabels,:);
                end
            end
                processedLabels{i,ii} = labels;
        end
        
        % Display progress after 300 files on screen.
        if ~mod(i,300)
            msg = sprintf('Processing data %3.2f%% complete', (i/numFiles)*100.0);
            fprintf(1,'%s',[tmpStr, msg]);
            tmpStr = repmat(sprintf('\b'), 1, length(msg));
        end
        processedData{i,1} = finalPC;
    end
    
    processedLabels = cell2table(processedLabels);
    numClasses = size(processedLabels,2);
    for j = 1:numClasses
        processedLabels.Properties.VariableNames{j} = classNames{j};
    end
    
    % Consider only class car for the detections.
    processedLabels = processedLabels(:,1);
    
     % Print completion message when done.
    msg = sprintf('Processing data 100%% complete');
    fprintf(1,'%s',[tmpStr, msg]);
end

function [gradients, loss, state] = modelGradients(net, pillarFeatures, pillarIndices, boxLabels, gridParams, anchorBoxes,...
                                                           executionEnvironment)

            % Extract the predictions from the network.
            YPredictions = cell(size(net.OutputNames));
            [YPredictions{:}, state] = forward(net,pillarIndices,pillarFeatures);

            % Generate target for predictions from the ground truth data.
            YTargets = generatePointPillarTargets(YPredictions, boxLabels, pillarIndices, gridParams, anchorBoxes);
            YTargets = cellfun(@ dlarray,YTargets,'UniformOutput', false);
            if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
                YTargets = cellfun(@ gpuArray,YTargets,'UniformOutput', false);
            end

            [angLoss, occLoss, locLoss, szLoss, hdLoss, clfLoss] = computePointPillarLoss(YPredictions, YTargets);

            % Compute the total loss.
            loss = angLoss + occLoss + locLoss + szLoss + hdLoss + clfLoss;

            % Compute the gradients of the learnables with regard to the loss.
            gradients = dlgradient(loss,net.Learnables);

        end

        function [pillarFeatures, pillarIndices, labels] = createBatchData(features, indices, groundTruthBoxes, groundTruthClasses, classNames)
        % Returns pillar features and indices combined along the batch dimension
        % and bounding boxes concatenated along batch dimension in labels.

            % Concatenate features and indices along batch dimension.
            pillarFeatures = cat(4, features{:,1});
            pillarIndices = cat(4, indices{:,1});

            % Get class IDs from the class names.
            classNames = repmat({categorical(classNames')}, size(groundTruthClasses));
            [~, classIndices] = cellfun(@(a,b)ismember(a,b), groundTruthClasses, classNames, 'UniformOutput', false);

            % Append the class indices and create a single array of responses.
            combinedResponses = cellfun(@(bbox, classid)[bbox, classid], groundTruthBoxes, classIndices, 'UniformOutput', false);
            len = max(cellfun(@(x)size(x,1), combinedResponses));
            paddedBBoxes = cellfun( @(v) padarray(v,[len-size(v,1),0],0,'post'), combinedResponses, 'UniformOutput',false);
            labels = cat(4, paddedBBoxes{:,1});
        end

        function lidarData = downloadWPIData(outputFolder, lidarURL)
        % Download the data set from the given URL into the output folder.

            lidarDataTarFile = fullfile(outputFolder,'WPI_LidarData.tar.gz');
            if ~exist(lidarDataTarFile, 'file')
                mkdir(outputFolder);

                disp('Downloading WPI Lidar driving data (760 MB)...');
                websave(lidarDataTarFile, lidarURL);
                untar(lidarDataTarFile,outputFolder);
            end

            % Extract the file.
            if ~exist(fullfile(outputFolder, 'WPI_LidarData.mat'), 'file')
                untar(lidarDataTarFile,outputFolder);
            end
            load(fullfile(outputFolder, 'WPI_LidarData.mat'),'lidarData');
            lidarData = reshape(lidarData,size(lidarData,2),1);
        end

        function net = downloadPretrainedPointPillarsNet(outputFolder, pretrainedNetURL)
        % Download the pretrained PointPillars detector.

            preTrainedMATFile = fullfile(outputFolder,'trainedPointPillarsNet.mat');
            preTrainedZipFile = fullfile(outputFolder,'trainedPointPillars.zip');

            if ~exist(preTrainedMATFile,'file')
                if ~exist(preTrainedZipFile,'file')
                    disp('Downloading pretrained detector (8.3 MB)...');
                    websave(preTrainedZipFile, pretrainedNetURL);
                end
                unzip(preTrainedZipFile, outputFolder);   
            end
            pretrainedNet = load(preTrainedMATFile);
            net = pretrainedNet.net;       
        end

        function lossPlotter = configureTrainingProgressPlotter(f)
        % The configureTrainingProgressPlotter function configures training
        % progress plots for various losses.
            figure(f);
            clf
            ylabel('Total Loss');
            xlabel('Iteration');
            lossPlotter = animatedline;
        end

        function retValue = checkForNaN(gradients,loss)
        % Based on experiments it is found that the last convolution head
        % 'occupancy|conv2d' contains NaNs as the gradients. This function checks
        % whether gradient values contain NaNs. Add other convolution
        % head values to the condition if NaNs are present in the gradients. 
            gradValue = gradients.Value((gradients.Layer == 'occupancy|conv2d') & (gradients.Parameter == 'Bias'));
            if (sum(isnan(extractdata(loss)),'all') > 0) || (sum(isnan(extractdata(gradValue{1,1})),'all') > 0)
                retValue = true;
            else
                retValue = false;
            end
        end
        
        function [processedData, processedLabels] = createFrontViewFromLidarData(ptCloudData, groundTruth, gridParams, fview)
    numFiles = size(ptCloudData,1);
    processedLabels = cell(size(groundTruth));
    processedData = cell(size(ptCloudData));

    theta = pi;
    rot = [cos(theta) sin(theta) 0; ...
          -sin(theta) cos(theta) 0; ...
                   0  0  1];
    trans = [0, 0, 0];
    tform = rigid3d(rot, trans);
    tmpStr = '';
    for i = 1:numFiles
        ptCloud = ptCloudData{i,1};
        ptCloud = pctransform(ptCloud,tform);

        % Get the indices of the point cloud that constitute the RoI
        % defined in the calibration parameters.
        [~, indices] = projectLidarPointsOnImage(ptCloud,fview.cameraParams, rigid3d(fview.tform.T));
        ptCloudTransformed = select(ptCloud, indices,'outputSize','full');
        ptCloudTransformed = pctransform(ptCloudTransformed, tform);

        % Set the limits for the point cloud.
        [row, column] = find( ptCloudTransformed.Location(:,:,1) < gridParams{1,2}{1} ...
                            & ptCloudTransformed.Location(:,:,1) > gridParams{1,1}{1} ...
                            & ptCloudTransformed.Location(:,:,2) < gridParams{1,2}{2} ...
                            & ptCloudTransformed.Location(:,:,2) > gridParams{1,1}{2} ...
                            & ptCloudTransformed.Location(:,:,3) < gridParams{1,2}{3} ...
                            & ptCloudTransformed.Location(:,:,3) > gridParams{1,1}{3});    
        ptCloudTransformed = select(ptCloudTransformed, row, column, 'OutputSize', 'full'); 
        finalPC = removeInvalidPoints(ptCloudTransformed);

        % Get the classes from the ground truth labels.
        classNames = groundTruth.Properties.VariableNames; 
        for ii = 1:numel(classNames)

            labels = groundTruth(i,classNames{ii}).Variables;
            labels = labels{1};
            if ~isempty(labels)

                % Get the label indices that are in the selected RoI.
                labelsIndices = labels(:,1) > gridParams{1,1}{1} ...
                            & labels(:,1) < gridParams{1,2}{1} ...
                            & labels(:,2) > gridParams{1,1}{2} ...
                            & labels(:,2) < gridParams{1,2}{2};
                labels = labels(labelsIndices,:);

                % Change the dimension of the ground truth object to fixed
                % value.
                carNewDim = repmat([3.9 1.6 1.56], size(labels,1),1);
                labels(:,4:6) = carNewDim;

                if ~isempty(labels)
                    % Find the number of points inside each ground truth
                    % label.
                    numPoints = arrayfun(@(x)(findPointsInsideCuboid(cuboidModel(labels(x,:)),finalPC)),...
                                (1:size(labels,1)).','UniformOutput',false);
                    posLabels = cellfun(@(x)(~isempty(x)),numPoints);
                    labels = labels(posLabels,:);
                end
            end
                processedLabels{i,ii} = labels;
        end
        
        % Display progress after 300 files on screen.
        if ~mod(i,300)
            msg = sprintf('Processing data %3.2f%% complete', (i/numFiles)*100.0);
            fprintf(1,'%s',[tmpStr, msg]);
            tmpStr = repmat(sprintf('\b'), 1, length(msg));
        end
        processedData{i,1} = finalPC;
    end
    
    processedLabels = cell2table(processedLabels);
    numClasses = size(processedLabels,2);
    for j = 1:numClasses
        processedLabels.Properties.VariableNames{j} = classNames{j};
    end
    
    % Consider only class car for the detections.
    processedLabels = processedLabels(:,1);
    
     % Print completion message when done.
    msg = sprintf('Processing data 100%% complete');
    fprintf(1,'%s',[tmpStr, msg]);
        end

        %% *Generate Training Data*
        % To save each point cloud as a mat file in the specified location
        function saveptCldToPCD(trainData,dataLocation)
    if ~exist(dataLocation,'dir')
        mkdir(dataLocation)
    end
    tmpStr = '';
    numFiles = size(trainData,1);
    for i = 1:numFiles
        ptCloud = trainData{i,1};
        pcFilePath = fullfile(dataLocation,sprintf('%06d.pcd',i));
        pcwrite(ptCloud,pcFilePath);
        
        % Display progress after 300 files on screen.
        if ~mod(i,300)
            msg = sprintf('Processing data %3.2f%% complete', (i/numFiles)*100.0);
            fprintf(1,'%s',[tmpStr, msg]);
            tmpStr = repmat(sprintf('\b'), 1, length(msg));
        end
        
    end
    
    % Print completion message when done.
    msg = sprintf('Processing data 100%% complete');
    fprintf(1,'%s',[tmpStr, msg]);
        end

        function groundTruthData = generateGTDataForAugmentation(processedData,groundTruth)
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
        
