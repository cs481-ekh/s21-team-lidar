outputFolder = fullfile('C:\Users\andres\Desktop\School STUFF\Spring21\s21-team-lidar\src\MatLab','Pandaset');

pretrainedNetURL = 'https://ssd.mathworks.com/supportfiles/lidar/data/trainedPointPillarsPandasetNet.zip';
matFile = helperDownloadPretrainedPointPillarsNet(outputFolder, pretrainedNetURL);
load(matFile,'net');


type('pointpillarsDetect.m')

%Define grid parameters.

xMin = -23.00;     % Minimum value along X-axis.
yMin = -23.00;  % Minimum value along Y-axis.
zMin = -23.00;    % Minimum value along Z-axis.
xMax = -9.00;   % Maximum value along X-axis.
yMax = -9.00;   % Maximum value along Y-axis.
zMax = -9.00;     % Maximum value along Z-axis.
xStep = 0.16;   % Resolution along X-axis.
yStep = 0.16;   % Resolution along Y-axis.
dsFactor = 2.0; % DownSampling factor.


%Calculate the dimensions for the pseudo-image.

Xn = round(((xMax - xMin) / xStep));
Yn = round(((yMax - yMin) / yStep));
gridParams = {{xMin,yMin,zMin},{xMax,yMax,zMax},{xStep,yStep,dsFactor},{Xn,Yn}};

%Define the pillar extraction parameters.

P = 12000; % Define number of prominent pillars.
N = 100;   % Define number of points per pillar.
%Calculate the number of network outputs.

networkOutputs = numel(net.OutputNames);
%Read a point cloud from the Pandaset data set [2] and convert it to M-by-4 format.

pc = pcread('C:\Users\andres\Desktop\School STUFF\Spring21\s21-team-lidar\src\MatLab\Pandaset\Lidar\0001.pcd');
intensity = reshape(pc.Intensity,[64,1856,1]);
ptCloud = cat(3,pc.Location,intensity);
%ptCloud = cat(3,pc.Location,intensity);

%Create pillar features and pillar indices from the point clouds using the createPillars helper function, attached to this example as a supporting file.

processedPtCloud = createPillars({ptCloud,'',''}, gridParams,P,N);
%Extract the pillar features and pillar indices.

pillarFeatures = dlarray(processedPtCloud{1,1},'SSCB');
pillarIndices = dlarray(processedPtCloud{1,2},'SSCB');
%Specify the class names.

classNames = {'car','truck'};
%Define the desired thresholds.

confidenceThreshold = 0.5;
overlapThreshold = 0.1;
%Use the detect method on the PointPillars network and display the results.

[bboxes,~,labels] = pointpillarsDetect(matFile, pillarFeatures, pillarIndices, gridParams, networkOutputs, confidenceThreshold, overlapThreshold, classNames);
bboxesCar = bboxes(labels == 'car',:);
bboxesTruck = bboxes(labels == 'truck',:);
%Display the detections on the point cloud.

helperDisplay3DBoxesOverlaidPointCloud(pc.Location, bboxesCar, 'green', bboxesTruck, 'magenta', 'Predicted bounding boxes');







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

function preTrainedMATFile = helperDownloadPretrainedPointPillarsNet(outputFolder, pretrainedNetURL)
% Download the pretrained PointPillars network.    
    if ~exist(outputFolder,'dir')
        mkdir(outputFolder);
    end
    preTrainedZipFile = fullfile(outputFolder,'trainedPointPillarsPandasetNet.zip');  
    preTrainedMATFile = fullfile(outputFolder,'trainedPointPillarsPandasetNet.mat');
    if ~exist(preTrainedMATFile,'file')
        if ~exist(preTrainedZipFile,'file')
            disp('Downloading pretrained detector (8.4 MB)...');
            websave(preTrainedZipFile, pretrainedNetURL);
        end
        unzip(preTrainedZipFile, outputFolder);   
    end   
end

function helperDisplay3DBoxesOverlaidPointCloud(ptCld, labelsCar, carColor, labelsTruck, truckColor, titleForFigure)
% Display the point cloud with different colored bounding boxes for different
% classes
    figure;
    ax = pcshow(ptCld);
    showShape('cuboid', labelsCar, 'Parent', ax, 'Opacity', 0.1, 'Color', carColor, 'LineWidth', 0.5);
    hold on;
    showShape('cuboid', labelsTruck, 'Parent', ax, 'Opacity', 0.1, 'Color', truckColor, 'LineWidth', 0.5);
    title(titleForFigure);
    zoom(ax,1.5);
end