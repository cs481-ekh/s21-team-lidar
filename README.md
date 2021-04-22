# Object Dection and Scene Stitching with Livox Horizon LiDar
### Introduction 
This program is designed to help gather lidar data from the Livox Horizon sensor, convert to 

### Requirements
  MatLab R2020a or higher

### Tools


### Development
 - LivoxLidarView.exe     - .lvx to .csv file
 - importPtCloudFromCSV.m - .csv to point clouds
 - organizedpointcloud.m  - point clouds to organized point clouds
 - pointcloudstitch.m     - stiching point clouds together
 - lidarMainGui.mlapp     - GUI app
 - pointpillarsDetect.m   - object detection script
### Sample Images

![CI](https://github.com/cs481-ekh/s21-team-lidar/workflows/CI/badge.svg)

## Instructions
### Gathering and Converting Data
Once your Livox Horizon sensor is plugged and connected to your device via ethernet, go to the Network and Sharing Center, go to properties for your ethernet adapter and  go to IPv4 settings. Choose manual IP, set your static IP to 192.168.1.50 and the subnet mask to 255.255.255.0.

Open Livox Viewer, if your settings are configured you should see the Horizon sensor in the Device manager. The sensor automatically turns on and will display lidar data. To begin recording Lidar data, click the play button in the upper left corner. You will begin recording a point cloud sequence until the pause button is pressed, at which point you will be asked to save an .lvx file.  
![image](https://user-images.githubusercontent.com/32054828/115180394-b22e9a00-a092-11eb-8619-534dab3a8d33.png)

Once you have a .lvx file, you must convert it to .csv for use in Matlab. From the top menu go to Tools -> File Converter and change to .lvx to .csv. Pick the Source File and name the converted target .csv file. Note: the file size of the .csv is considerably more than the .lvx.   
![image](https://user-images.githubusercontent.com/32054828/115180551-105b7d00-a093-11eb-8586-31aee8056c87.png)

### Using the program
After opening the program, you will be greeted with this GUI.  
![image](https://user-images.githubusercontent.com/32054828/115182049-7ac1ec80-a096-11eb-90b6-15ae1e5779ab.png)  

Click Select Point Cloud file (.csv) to open a .csv point cloud file.

To created a stitched scene, click Stitch Point Clouds. This will load points clouds from the .csv and merge them together into a complete scene. This will take some time depending on the size of the .csv file. When it's done, a Matlab figuring rendering the enviroment will pop up. 
![image](https://user-images.githubusercontent.com/32054828/115182531-59153500-a097-11eb-847f-ce1611f0cefd.png)

To detect objects in the point cloud sequence, click Detect Objects. This will load point clouds from the .csv, convert them to organized point clouds, write them to .pcd files, download the neural network and detect objects via Machine learning. This will take considerable time and the resulting files will take up some hard drive space. When finished, a Matlab figure of detected objects in bounding boxes will appear.

![image](https://user-images.githubusercontent.com/32054828/115302119-4c3c2400-a11f-11eb-8896-75c0db79005f.png)

