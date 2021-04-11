function pass = unitTests(filepath)
    pass = true;
    disp("Testing lvxsample2-1");
    name = "lvxsample2-1";
    csv = filepath + "\" +name+".csv";
    pass = pass & testImportPtCloudFromCSV(csv);
    pass = pass & testOrganizedPointCloud(csv, filepath, name);
    pass = pass & testPointCloudStitch(csv);
    pass = pass & testPointPillars(filepath +"\opd_out");
    if pass
        disp("All tests passing");
    else
        disp("Test failed");
    end
end

function pass = testImportPtCloudFromCSV(filepath)
    disp("Testing importPtCloudFromCSV");
    tic
    try
    pointCloud = importPtCloudFromCSV(filepath);
    catch ME 
        time = toc;
        disp("Call to importPtCloudFromCSV took: " + time + " seconds")
        disp("importPtCloudFromCSV failed due to exception: " + ME.message);
        pass = false;
        return;
    end
    time = toc;
    disp("importPtCloudFromCSV took: " + time + " seconds");
    pcshow(pointCloud{1});
    pass = input("Does point cloud pass test? [true/false]");
    if pass 
        disp("testImportPtCloudFromCSV : " + filepath + " passed!");
    else 
        disp("testImportPtCloudFromCSV : " + filepath + " failed!");
    end
end

function pass = testOrganizedPointCloud(csv, outputPath, name)
    disp("Testing organizedpointcloud");
    tic 
    try
    pass = organizedpointcloud(csv, outputPath+"\opd_out", name);
    catch ME 
        time = toc;
        disp("Call to organizedpointcloud took: " + time + " seconds")
        disp("organizedpointcloud failed due to exception: " + ME.message);
        pass = false;
        return;
    end
    time = toc;
    disp("organizedpointcloud took: " + time + " seconds");
    pcshow(pcread(outputPath+"\opd_out\"+ name +"_00001.pcd"));
end

function pass = testPointCloudStitch(filepath)
    disp("Testing pointCloudStitch");
    tic 
    try
    stitchedPointCloud = pointCloudStitch(filepath);
    catch ME 
        time = toc;
        disp("Call to pointCloudStitch took: " + time + " seconds")
        disp("pointCloudStitch failed due to exception: " + ME.message);
        pass = false;
        return;
    end
    time = toc;
    disp("pointCloudStitch took: " + time + " seconds");
    pcshow(stitchedPointCloud);
    pass = input("Does point cloud pass test? [true/false]");
    if pass 
        disp("pointCloudStitch : " + filepath + " passed!");
    else 
        disp("pointCloudStitch : " + filepath + " failed!");
    end
end

function pass = testPointPillars(filepath)
    disp("Testing point pillars model");
    tic 
    try
    pp();
    catch ME 
        time = toc;
        disp("Call to pp took: " + time + " seconds")
        disp("pp failed due to exception: " + ME.message);
        pass = false;
        return;
    end
    time = toc;
    disp("pp took: " + time + " seconds");
end