
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


%%PREPROCESS%%

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


%Display the cropped point cloud with ground truth box labels 
pc = croppedPointCloudObj{1,1};
gtLabelsCar = processedLabels.Car{1};
gtLabelsTruck = processedLabels.Truck{1};

helperDisplay3DBoxesOverlaidPointCloud(pc.Location,gtLabelsCar,...
   'green',gtLabelsTruck,'magenta','Cropped Point Cloud');
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%HELPERS%%%%%%%%%%%%%%%%%%%
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