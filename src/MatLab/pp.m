
function [croppedPC, croppedLabels] = pp(pcdFolder, groundTruthPath, trainingFolder)
    
f = waitbar(0,'Loading Point Cloud Data...');
outputFolder = fullfile(trainingFolder);
lidarURL = ['https://ssd.mathworks.com/supportfiles/lidar/data/' ...
    'Pandaset_LidarData.tar.gz'];
helperDownloadPandasetData(outputFolder,lidarURL);

pretrainedNetURL = ['https://ssd.mathworks.com/supportfiles/lidar/data/' ...
    'trainedPointPillarsPandasetNet.zip'];

doTraining = false;
if ~doTraining
    helperDownloadPretrainedPointPillarsNet(outputFolder,pretrainedNetURL);
end


%Load Data
%Create a file datastore to load the PCD files from the specified path using the pcread function.

path = fullfile(pcdFolder);
lidarData = fileDatastore(path,'ReadFcn',@(x) pcread(x));

%Turn groundtruth into timetable and load the 3-D bounding
% Box labels of the car(col2) and truck(col3) objects.

gtPath = fullfile(groundTruthPath);
data = load(gtPath,'lidarGtLabels');
Labels = timetable2table(data.lidarGtLabels);
boxLabels = Labels(:,2:3 );

%Display the full-view point cloud.

% figure
% ptCld = read(lidarData);
% ax = pcshow(ptCld.Location);
% set(ax,'XLim',[-50 50],'YLim',[-40 40]);
% zoom(ax,2.5);
% axis off;

%uncomment to reset
reset(lidarData);


%%PREPROCESS%%%
%%%%%%%%%%%%%%%%

xMin = 0.0;     % Minimum value along X-axis.
yMin = -39.68;  % Minimum value along Y-axis.
zMin = -5.0;    % Minimum value along Z-axis.
xMax = 69.12;   % Maximum value along X-axis.
yMax = 39.68;   % Maximum value along Y-axis.
zMax = 5.0;     % Maximum value along Z-axis.
xStep = 0.16;   % Resolution along X-axis.
yStep = 0.16;   % Resolution along Y-axis.
dsFactor = 2.0; % Downsampling factor.

% Calculate the dimensions for the pseudo-image.
Xn = round(((xMax - xMin) / xStep));
Yn = round(((yMax - yMin) / yStep));

% Define the pillar extraction parameters.
gridParams = {{xMin,yMin,zMin},{xMax,yMax,zMax},{xStep,yStep,dsFactor},{Xn,Yn}};
waitbar(.1,'Processing Point Cloud Data...');

%Select the box labels that are inside the ROI specified by gridParams.

[croppedPointCloudObj,processedLabels] = cropFrontViewFromLidarData(...
    lidarData,boxLabels,gridParams);

croppedPC = croppedPointCloudObj;
croppedLabels = processedLabels;


%Uncomment to display the cropped point cloud with ground truth box labels 

% pc = croppedPointCloudObj{1,1};
% gtLabelsCar = processedLabels.Car{1};
% gtLabelsTruck = processedLabels.Truck{1};
% 
% helperDisplay3DBoxesOverlaidPointCloud(pc.Location,gtLabelsCar,...
%    'green',gtLabelsTruck,'magenta','Cropped Point Cloud');
% 
% reset(lidarData);



% Create Datastore Objects for Training
% Split the data set into training and test sets. Select 70% of the data for training the network and the rest for evaluation.
% 
rng(1);
shuffledIndices = randperm(size(processedLabels,1));
idx = floor(0.7 * length(shuffledIndices));

trainData = croppedPointCloudObj(shuffledIndices(1:idx),:);
testData = croppedPointCloudObj(shuffledIndices(idx+1:end),:);

trainLabels = processedLabels(shuffledIndices(1:idx),:);
testLabels = processedLabels(shuffledIndices(idx+1:end),:);



%Save Training Data as PCD
writeFiles = true; %false if trainingData exists
dataLocation = fullfile(outputFolder,'InputData');
[trainData,trainLabels] = saveptCldToPCD(trainData,trainLabels,...
    dataLocation,writeFiles);


% Create a file datastore using fileDatastore to load PCD files using the pcread function.

lds = fileDatastore(dataLocation,'ReadFcn',@(x) pcread(x));
% Create a box label datastore using boxLabelDatastore for loading the 3-D bounding box labels.

bds = boxLabelDatastore(trainLabels);
% Use the combine function to combine the point clouds and 3-D bounding box labels into a single datastore for training.

cds = combine(lds,bds);

%Data AUGMENTATION%
%%%%%%%%%%%%%%%%%%%

augData = read(cds);
augptCld = augData{1,1};
augLabels = augData{1,2};
augClass = augData{1,3};

labelsCar = augLabels(augClass=='Car',:);
labelsTruck = augLabels(augClass=='Truck',:);


%UsegenerateGTDataForAugmentation to extract all the ground truth bounding boxes from the training data.

gtData = generateGTDataForAugmentation(trainData,trainLabels);


%randomly add a fixed number of car and truck class objects to every point cloud.

samplesToAdd = struct('Car',10,'Truck',10);
cdsAugmented = transform(cds,@(x) groundTruthDataAugmenation(x,gtData,samplesToAdd));

% Apply to each point cloud
% Random flipping along the x-axis
% Random scaling by 5 percent
% Random rotation along the z-axis from [-pi/4, pi/4]
% Random translation by [0.2, 0.2, 0.1] meters along the x-, y-, and z-axis respectively

cdsAugmented = transform(cdsAugmented,@(x) augmentData(x));



%extract pillar information%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%convert the 3-D point clouds to 2-D representation. Use the transform function 

% Define number of prominent pillars.
P = 12000; 

% Define number of points per pillar.
N = 100;   
cdsTransformed = transform(cdsAugmented,@(x) createPillars(x,gridParams,P,N));


%Define PILLAR NETWORK%
%%%%%%%%%%%%%%%%%%%%%%%


%Define the anchor box dimensions based on the classes to detect.

anchorBoxes = calculateAnchorsPointPillars(trainLabels);
numAnchors = size(anchorBoxes,2);
classNames = trainLabels.Properties.VariableNames;
numClasses = numel(classNames);


%create the PointPillars network

lgraph = pointpillarNetwork(numAnchors,gridParams,P,N,numClasses);

%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%Training Options%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%

numEpochs = 60;
miniBatchSize = 2;
learningRate = 0.0002;
learnRateDropPeriod = 15;
learnRateDropFactor = 0.8;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.999;
trailingAvg = [];
trailingAvgSq = [];


%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%Train Model%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%

executionEnvironment = "auto";
if canUseParallelPool
    dispatchInBackground = true;
else
    dispatchInBackground = false;
end

mbq = minibatchqueue(...
    cdsTransformed,3,...
    "MiniBatchSize",miniBatchSize,...
    "OutputEnvironment",executionEnvironment,...
    "MiniBatchFcn",@(features,indices,boxes,labels) ...
    helperCreateBatchData(features,indices,boxes,labels,classNames),...
    "MiniBatchFormat",["SSCB","SSCB",""],...
    "DispatchInBackground",true);



if doTraining
    % Convert layer graph to dlnetwork.
    net = dlnetwork(lgraph);
    
    % Initialize plot.
    fig = figure;
    lossPlotter = helperConfigureTrainingProgressPlotter(fig);    
    iteration = 0;
       
    % Custom training loop.
    for epoch = 1:numEpochs
        
        % Reset datastore.
        reset(mbq);
        while(hasdata(mbq))
            iteration = iteration + 1;
            
            % Read batch of data.
            [pillarFeatures,pillarIndices,boxLabels] = next(mbq);
                        
            % Evaluate the model gradients and loss using dlfeval 
            % and the modelGradients function.
            [gradients,loss,state] = dlfeval(@modelGradients,net,...
                pillarFeatures,pillarIndices,boxLabels,gridParams,...
                anchorBoxes,numClasses,executionEnvironment);
            
            % Do not update the network learnable parameters if NaN values
            % are present in gradients or loss values.
            if helperCheckForNaN(gradients,loss)
                continue;
            end
                    
            % Update the state parameters of dlnetwork.
            net.State = state;
            
            % Update the network learnable parameters using the Adam
            % optimizer.
            [net.Learnables,trailingAvg,trailingAvgSq] = ...
                adamupdate(net.Learnables,gradients,trailingAvg,...
                trailingAvgSq,iteration,learningRate,...
                gradientDecayFactor,squaredGradientDecayFactor);
            
            % Update training plot with new points.         
            addpoints(lossPlotter,iteration,double(gather(extractdata(loss))));
            title("Training Epoch " + epoch +" of " + numEpochs);
            drawnow;
        end
        
        % Update the learning rate after every learnRateDropPeriod.
        if mod(epoch,learnRateDropPeriod) == 0
            learningRate = learningRate * learnRateDropFactor;
        end
    end
else
    preTrainedMATFile = fullfile(outputFolder,'trainedPointPillarsPandasetNet.mat');
    pretrainedNetwork = load(preTrainedMATFile,'net');
    net = pretrainedNetwork.net;
end



%%%%%%%%%%%%%%%%%%%%%%%%%
%%Generate Detections%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%


ptCloud = testData{45,1};
gtLabels = testLabels(45,:);

% The generatePointPillarDetections function detects the 
% bounding boxes, and scores for a given point cloud.
confidenceThreshold = 0.5;
overlapThreshold = 0.1;
[box,score,labels] = generatePointPillarDetections(net,ptCloud,anchorBoxes,...
    gridParams,classNames,confidenceThreshold,overlapThreshold,P,N,...
    executionEnvironment);

boxlabelsCar = box(labels'=='Car',:);
boxlabelsTruck = box(labels'=='Truck',:);

% Uncomment to display the predictions on the point cloud.

% helperDisplay3DBoxesOverlaidPointCloud(ptCloud.Location,boxlabelsCar,'green',...
%     boxlabelsTruck,'magenta','Predicted Bounding Boxes');




%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%Evaluate Model%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%


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
    
    [box,score,labels] = generatePointPillarDetections(net,ptCloud,anchorBoxes,...
        gridParams,classNames,confidenceThreshold,overlapThreshold,...
        P,N,executionEnvironment);
 
    % Convert the detected boxes to rotated rectangle format.
    if ~isempty(box)
        detectionResults.Boxes{num} = box(:,[1,2,4,5,7]);
    else
        detectionResults.Boxes{num} = box;
    end
    detectionResults.Scores{num} = score;
    detectionResults.Labels{num} = labels;
end

metrics = evaluateDetectionAOS(detectionResults,groundTruthData,...
    nmsPositiveIoUThreshold);
disp(metrics(:,1:2))








end







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%HELPERS%%%%%%%%%%%%%%%%%%%
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

function [gradients,loss,state] = modelGradients(net,pillarFeatures,...
    pillarIndices,boxLabels,gridParams,anchorBoxes,...
    numClasses,executionEnvironment)
      
    numAnchors = size(anchorBoxes,2);
    
    % Extract the predictions from the network.
    YPredictions = cell(size(net.OutputNames));
    [YPredictions{:},state] = forward(net,pillarIndices,pillarFeatures);
    
    % Generate target for predictions from the ground truth data.
    YTargets = generatePointPillarTargets(YPredictions,boxLabels,pillarIndices,...
        gridParams,anchorBoxes,numClasses);
    YTargets = cellfun(@ dlarray,YTargets,'UniformOutput',false);
    if (executionEnvironment=="auto" && canUseGPU) || executionEnvironment=="gpu"
        YTargets = cellfun(@ gpuArray,YTargets,'UniformOutput',false);
    end
     
    [angLoss,occLoss,locLoss,szLoss,hdLoss,clfLoss] = ...
        computePointPillarLoss(YPredictions,YTargets,gridParams,...
        numClasses,numAnchors);
    
    % Compute the total loss.
    loss = angLoss + occLoss + locLoss + szLoss + hdLoss + clfLoss;
    
    % Compute the gradients of the learnables with regard to the loss.
    gradients = dlgradient(loss,net.Learnables);
 
end

function [pillarFeatures,pillarIndices,labels] = helperCreateBatchData(...
    features,indices,groundTruthBoxes,groundTruthClasses,classNames)
% Return pillar features and indices combined along the batch dimension
% and bounding boxes concatenated along batch dimension in labels.
    
    % Concatenate features and indices along batch dimension.
    pillarFeatures = cat(4,features{:,1});
    pillarIndices = cat(4,indices{:,1});
    
    % Get class IDs from the class names.
    classNames = repmat({categorical(classNames')},size(groundTruthClasses));
    [~,classIndices] = cellfun(@(a,b)ismember(a,b),groundTruthClasses,...
        classNames,'UniformOutput',false);
    
    % Append the class indices and create a single array of responses.
    combinedResponses = cellfun(@(bbox,classid) [bbox,classid],...
        groundTruthBoxes,classIndices,'UniformOutput',false);
    len = max(cellfun(@(x)size(x,1),combinedResponses));
    paddedBBoxes = cellfun(@(v) padarray(v,[len-size(v,1),0],0,'post'),...
        combinedResponses,'UniformOutput',false);
    labels = cat(4,paddedBBoxes{:,1});
end

function helperDownloadPandasetData(outputFolder,lidarURL)
% Download the data set from the given URL to the output folder.

    lidarDataTarFile = fullfile(outputFolder,'Pandaset_LidarData.tar.gz');
    
    if ~exist(lidarDataTarFile,'file')
        mkdir(outputFolder);
        
        disp('Downloading PandaSet Lidar driving data (5.2 GB)...');
        websave(lidarDataTarFile,lidarURL);
        untar(lidarDataTarFile,outputFolder);
    end
    
    % Extract the file.
    if (~exist(fullfile(outputFolder,'Lidar'),'dir'))...
            &&(~exist(fullfile(outputFolder,'Cuboids'),'dir'))
        untar(lidarDataTarFile,outputFolder);
    end

end

function helperDownloadPretrainedPointPillarsNet(outputFolder,pretrainedNetURL)
% Download the pretrained PointPillars network.

    preTrainedMATFile = fullfile(outputFolder,'trainedPointPillarsPandasetNet.mat');
    preTrainedZipFile = fullfile(outputFolder,'trainedPointPillarsPandasetNet.zip');
    
    if ~exist(preTrainedMATFile,'file')
        if ~exist(preTrainedZipFile,'file')
            disp('Downloading pretrained detector (8.4 MB)...');
            websave(preTrainedZipFile,pretrainedNetURL);
        end
        unzip(preTrainedZipFile,outputFolder);   
    end       
end

function lossPlotter = helperConfigureTrainingProgressPlotter(f)
% This function configures training progress plots for various losses.
    figure(f);
    clf
    ylabel('Total Loss');
    xlabel('Iteration');
    lossPlotter = animatedline;
end

function retValue = helperCheckForNaN(gradients,loss)
% The last convolution head 'occupancy|conv2d' is known to contain NaNs 
% the gradients. This function checks whether gradient values contain 
% NaNs. Add other convolution head values to the condition if NaNs 
% are present in the gradients. 
    gradValue = gradients.Value((gradients.Layer == 'occupancy|conv2d') & ...
        (gradients.Parameter == 'Bias'));
    if (sum(isnan(extractdata(loss)),'all') > 0) || ...
            (sum(isnan(extractdata(gradValue{1,1})),'all') > 0)
        retValue = true;
    else
        retValue = false;
    end
end

function [croppedPointCloudObj, processedLabels] = cropFrontViewFromLidarData(lidarData, boxLabels, gridParams)
% This function crops the front view from the input full-view point cloud
% and also processes the corresponding box labels according to the 
% specified grid parameters.

    tmpStr = '';
    numFiles = size(boxLabels,1);
    
    processedLabels = cell(size(boxLabels));
    croppedPointCloudObj = cell(size(numFiles));

    % Get the classes from the ground truth labels.
    classNames = boxLabels.Properties.VariableNames
    
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


function helperDisplay3DBoxesOverlaidPointCloud(ptCld,labelsCar,carColor,...
    labelsTruck,truckColor,titleForFigure)
% Display the point cloud with different colored bounding boxes for different
% classes.
    figure;
    ax = pcshow(ptCld);
    showShape('cuboid',labelsCar,'Parent',ax,'Opacity',0.1,...
        'Color',carColor,'LineWidth',0.5);
    hold on;
    showShape('cuboid',labelsTruck,'Parent',ax,'Opacity',0.1,...
        'Color',truckColor,'LineWidth',0.5);
    title(titleForFigure);
    zoom(ax,1.5);
end


function [ptCld, ptLabels] = saveptCldToPCD(ptCld, ptLabels, dataLocation, writeFiles)
% This function saves the required point clouds in the specified location 

    if ~exist(dataLocation, 'dir')
        mkdir(dataLocation)
    end
    
    tmpStr = '';
    numFiles = size(ptLabels,1);
    ind = [];
    
    for i = 1:numFiles
        processedData = ptCld{i,1};
        
        % Skip if the processed point cloud is empty
        if(isempty(processedData.Location))
            ind = [ind, i];
            continue;
        end
        
        if(writeFiles)
            pcFilePath = fullfile(dataLocation, sprintf('%06d.pcd',i));
            pcwrite(processedData, pcFilePath);
        end
      
        % Display progress after 300 files on screen.
        if ~mod(i,300)
            msg = sprintf('Processing data %3.2f%% complete', (i/numFiles)*100.0);
            fprintf(1,'%s',[tmpStr, msg]);
            tmpStr = repmat(sprintf('\b'), 1, length(msg));
        end
    end
    
    ptCld(ind,:) = [];
    ptLabels(ind,:) = [];
    
    % Print completion message when done.
    msg = sprintf('Processing data 100%% complete');
    fprintf(1,'%s',[tmpStr, msg]);
end


