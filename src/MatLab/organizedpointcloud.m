
function success = organizedpointcloud(inputPath, outputPath, filename)
    x = importPtCloudFromCSV(inputPath);

    width = 1024*5;
    height = 64*5;


    totalClouds = size(x,1);
    duration = seconds(totalClouds/10);


    for cloudnum = 1.0:1:size(x,1)
        points = zeros(height,width,3);
        intensityMap = zeros(height,width);
        y = x{cloudnum,1};
        XResolution = abs(y.XLimits(2) - y.XLimits(1)) / width;
        YResolution = abs(y.YLimits(2) - y.YLimits(1)) / height;
        for v = 1:1.0:size(y.Location,1)
            sample = y.Location(v,:);
            N = round(abs(sample(1) - y.XLimits(1)) / XResolution);
            if (N == 0)
                N = 1;
            end
            M = round(abs(sample(2) - y.YLimits(1)) / YResolution);
            if (M == 0)
                M = 1;
            end
            points(M,N,1) = sample(1);
            points(M,N,2) = sample(2);
            points(M,N,3) = sample(3);
            intensityMap(M,N) = y.Intensity(v);
        end 
        ptCloud = pointCloud(points, 'Intensity', intensityMap);
        pcshow(ptCloud);
        x{cloudnum,1} = [];
        %if (mod((cloudnum - 13),11) ~= 0)
        name = sprintf( '%05d', cloudnum );
        ptCloudToPCD(ptCloud, outputPath, filename +"_" + name);
        %end

    end
    success = 1;

end


