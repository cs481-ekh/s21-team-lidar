function ptCloudToPCD(ptCld, path, name)
%ptcellArrayToPCD Summary of this function goes here
%   Converts a cell array of point clouds to a PCD sequence in a new folder
%   ptcell : 
mkdir(path);
pcwrite(ptCld, path + '\' + name + '.pcd', 'Encoding', 'compressed')


end
