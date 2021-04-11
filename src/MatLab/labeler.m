% lidarDir = fullfile(matlabroot,'toolbox','lidar','lidardata','lidarLabeler');
% addpath(lidarDir)
% load('lidarLabelerGTruth.mat') %labels definitons


%automation algorithm
projectFolder = fullfile('C:\Users\andres\Desktop\School STUFF\Spring21\s21-team-lidar\src\MatLab');%change this to working directory
automationFolder = fullfile('+vision','+labeler');

if ( not(isfolder(automationFolder)))
    mkdir(projectFolder,automationFolder)
    addpath(automationFolder)
end


% ldc = labelDefinitionCreatorLidar(lidarLabelerGTruth.LabelDefinitions)

% addAttribute(ldc,'car','Color','List',{'Red','Green','Blue'})
% 
% ldc
objectLoc = table(gTruth.DataSource.Source,gTruth.LabelData.pedestrian);
% Training Option
options = trainingOptions('sgdm', ...
        'MiniBatchSize', 2, ....
        'InitialLearnRate',1e-4, ...
        'MaxEpochs',5,...
        'CheckpointPath', tempdir, ...
        'executionEnvironment','gpu',...
        'Shuffle','every-epoch');
    
    
    
 [detector, info] = trainFasterRCNNObjectDetector(ObjectLoc, "alexnet", options);