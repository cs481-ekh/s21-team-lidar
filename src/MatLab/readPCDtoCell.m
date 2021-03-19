function ptCell = readPCDtoCell(folderpath, name) 
    i = 1;
    if ~isfile(folderpath+"\"+name+"_"+i+".pcd")
        ME = MException('readPCDtoCell:FileNotFound', ...
        'Point Cloud File %s\\%s_%d.pcd',folderpath, name, i);
        throw(ME);
    end
    while isfile(folderpath+"\"+name+"_"+i+".pcd") 
        ptCell{i,1} = pcread(folderpath+"\"+name+"_"+i+".pcd");
        i = i +1;
    end
end