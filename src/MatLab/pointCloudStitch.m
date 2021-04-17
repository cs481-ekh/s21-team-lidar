function ptCloudScene = pointCloudStitch(filepath)
    f = waitbar(0,'Loading Point Clouds...');
    pointClouds = importPtCloudFromCSV(filepath);
    %pcshow(pointClouds{1})
    %load(dataFile);
    waitbar(0, f,'Stitching Point Clouds...');
    % Extract two consecutive point clouds and use the first point cloud as
    % reference.
    ptCloudRef = pointClouds{1};
    ptCloudCurrent = pointClouds{2};
    
    gridSize = 0.1;
    fixed = pcdownsample(ptCloudRef, 'gridAverage', gridSize);
    moving = pcdownsample(ptCloudCurrent, 'gridAverage', gridSize);

    tform = pcregistericp(moving, fixed, 'Metric','pointToPlane','Extrapolate', true);
    ptCloudAligned = pctransform(ptCloudCurrent,tform);

    mergeSize = 0.015; 
    ptCloudScene = pcmerge(ptCloudRef, ptCloudAligned, mergeSize);

    % % Visualize the input images.
    % figure
    % subplot(2,2,1)
    % imshow(ptCloudRef.Color)
    % title('First input image','Color','w')
    % drawnow
    % 
    % subplot(2,2,3)
    % imshow(ptCloudCurrent.Color)
    % title('Second input image','Color','w')
    % drawnow

    % % Visualize the world scene.
    % subplot(2,2,[2,4])
    % pcshow(ptCloudScene, 'VerticalAxis','Y', 'VerticalAxisDir', 'Down')
    % title('Initial world scene')
    % xlabel('X (m)')
    % ylabel('Y (m)')
    % zlabel('Z (m)')
    % drawnow

    % Store the transformation object that accumulates the transformation.
    accumTform = tform; 

    %hAxes = pcshow(ptCloudScene, 'VerticalAxis','Y', 'VerticalAxisDir', 'Down'); 
    % Set the axes property for faster rendering
    %hAxes.CameraViewAngleMode = 'auto';
%   hScatter = hAxes.Children;

    for i = 3:1:length(pointClouds)
        ptCloudCurrent = pointClouds{i};

        % Use previous moving point cloud as reference.
        fixed = moving;
        moving = pcdownsample(ptCloudCurrent, 'gridAverage', gridSize);

        % Apply ICP registration.
        tform = pcregistericp(moving, fixed, 'Metric','pointToPlane','Extrapolate', true);

        % Transform the current point cloud to the reference coordinate system
        % defined by the first point cloud.
        accumTform = affine3d(tform.T * accumTform.T);
        ptCloudAligned = pctransform(ptCloudCurrent, accumTform);

        % Update the world scene.
        ptCloudScene = pcmerge(ptCloudScene, ptCloudAligned, mergeSize);

%         % Visualize the world scene.
%         hScatter.XData = ptCloudScene.Location(:,1);
%         hScatter.YData = ptCloudScene.Location(:,2);
%         hScatter.ZData = ptCloudScene.Location(:,3);
%     `   hScatter.CData = ptCloudScene.Color;
%         drawnow('limitrate')
        waitbar(i/length(pointClouds), f,'Stitching Point Clouds...');
    end

    % During the recording, the Kinect was pointing downward. To visualize the
    % result more easily, let's transform the data so that the ground plane is
    % parallel to the X-Z plane.
    % angle = -pi/10;
    % A = [1,0,0,0;...
    %      0, cos(angle), sin(angle), 0; ...
    %      0, -sin(angle), cos(angle), 0; ...
    %      0 0 0 1];
    % ptCloudScene = pctransform(ptCloudScene, affine3d(A));
%    pcshow(ptCloudScene, 'VerticalAxis','Y', 'VerticalAxisDir', 'Down', ...
%              'Parent', hAxes)
    % title('Updated world scene')
    % xlabel('X (m)')
    % ylabel('Y (m)')
    % zlabel('Z (m)')
    % pcshow(ptCloudScene);
     delete(f);
end