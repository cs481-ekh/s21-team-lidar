function ptcellArrayToPCD(ptCell, path, name)
%ptcellArrayToPCD Summary of this function goes here
%   Converts a cell array of point clouds to a PCD sequence in a new folder
%   ptcell : 
mkdir(path);
for v = 1:size(ptCell)
    pcwrite(ptCell{v,1}, path + '\' + name + '_' + v + '.pcd', 'Encoding', 'compressed')
end

end

