#include <iostream>
#include <Eigen/Eigen>
#include <vector>
#include <string>
#include <cmath>
#include "pcl_ros_utilities/pcl_ros_utilities.hpp"

vector<vector<float> > PCLUtilities::pointCloud2ToVec(const pcl::PCLPointCloud2& p)
{
    vector<vector<float> > v; 
    for(int i=0;i<p.row_step;i+=p.point_step)
    {
        vector<float> t;
        for(int j=0;j<3;j++)
        {
            if(p.fields[j].count==0)
            {
                continue;
            }
            float x;
            memcpy(&x,&p.data[i+p.fields[j].offset],sizeof(float));
            t.push_back(x);
        }
        if(p.point_step>16)
        {
            float rgb;
            memcpy(&rgb,&p.data[i+p.fields[3].offset],sizeof(float));
            vector<int> c = splitRGBData(rgb);
            for(int k=0;k<3;k++)
                t.push_back(float(c[k]));
        }        	
        v.push_back(t);
    }
    return v;
}

pcl::PointCloud<pcl::PointXYZ> PCLUtilities::pointCloud2ToPclXYZ(const pcl::PCLPointCloud2& p)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;
    for(int i=0;i<p.row_step;i+=p.point_step)
    {
        vector<float> t;
        for(int j=0;j<3;j++)
        {
            if(p.fields[j].count==0)
            {
                continue;
            }
            float x;
            memcpy(&x,&p.data[i+p.fields[j].offset],sizeof(float));
            t.push_back(x);
        }
        cloud.points.push_back(PointXYZ(t[0],t[1],t[2]));
    }
    return cloud;
}	

pcl::PointCloud<pcl::PointXYZRGB> PCLUtilities::pointCloud2ToPclXYZRGB(const pcl::PCLPointCloud2& p)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    for(int i=0;i<p.row_step;i+=p.point_step)
    {
        vector<float> t;
        for(int j=0;j<3;j++)
        {
            if(p.fields[j].count==0)
            {
                continue;
            }
            float x;
            memcpy(&x,&p.data[i+p.fields[j].offset],sizeof(float));
            t.push_back(x);
        }
        float rgb_data;
        memcpy(&rgb_data,&p.data[i+p.fields[3].offset],sizeof(float));
        vector<int> c = splitRGBData(rgb_data);    
        pcl::PointXYZRGB point;
        point.x = t[0];
        point.y = t[1];
        point.z = t[2];
        uint32_t rgb = (static_cast<uint32_t>(c[0]) << 16 |
                static_cast<uint32_t>(c[1]) << 8 | static_cast<uint32_t>(c[2]));
        point.rgb = *reinterpret_cast<float*>(&rgb);
        cloud.push_back(point);    	
    }
    return cloud;
}	

vector<vector<float> > PCLUtilities::pointCloud2ToVec(const sensor_msgs::PointCloud2& p)
{
    vector<vector<float> > v; 
    for(int i=0;i<p.row_step;i+=p.point_step)
    {
        vector<float> t;
        for(int j=0;j<3;j++)
        {
            if(p.fields[j].count==0)
            {
                continue;
            }
            float x;
            memcpy(&x,&p.data[i+p.fields[j].offset],sizeof(float));
            t.push_back(x);
        }
        if(p.point_step>16)
        {
            float rgb;
            memcpy(&rgb,&p.data[i+p.fields[3].offset],sizeof(float));
            vector<int> c = splitRGBData(rgb);
            for(int k=0;k<3;k++)
                t.push_back(float(c[k]));
        }        	
        v.push_back(t);
    }
    return v;
}	

void PCLUtilities::xyzToPcd(const string &input_file, const string &output_file)
{
    ifstream fs;
    fs.open (input_file.c_str (), ios::binary);
    if (!fs.is_open () || fs.fail ())
    {
        PCL_ERROR ("Could not open file '%s'! Error : %s\n", input_file.c_str (), strerror (errno)); 
        fs.close ();
        return ;
    }
    string line;
    vector<string> st;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    while (!fs.eof ())
    {
        getline (fs, line);
        // Ignore empty lines
        if (line.empty())
            continue;
        boost::trim (line);
        boost::split (st, line, boost::is_any_of ("\t\r "), boost::token_compress_on);
        if (st.size () != 3)
            continue;
        cloud.push_back (PointXYZ (float (atof (st[0].c_str ())), float (atof (st[1].c_str ())), float (atof (st[2].c_str ()))));
    }
    fs.close ();
    cloud.width = uint32_t (cloud.size ()); cloud.height = 1; cloud.is_dense = true;
    // Convert to PCD and save
    PCDWriter w;
    w.writeBinaryCompressed (output_file, cloud);
}

pcl::PCLPointCloud2 PCLUtilities::pcdToPointCloud2(const std::string &filename)
{
    std::cout<<"Loading ";
    pcl::PCLPointCloud2 cloud;
    pcl::PLYReader reader;
    if (reader.read (filename, cloud) < 0)
        PCL_ERROR ("Unable to read the file. \n"); 
    return cloud;
}

pcl::PointCloud<pcl::PointXYZRGB> PCLUtilities::makePointCloud(Mat& color_image, Mat& depth_image, Eigen::VectorXd& K, std::string& frame_id)
{
    double fx=K[0],cx=K[2],fy=K[4],cy=K[5];
    pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
    Size s = depth_image.size();
    int width = s.width;
    int height = s.height;
    pcl_cloud.width = width*height;
    pcl_cloud.height = 1;
    pcl_cloud.is_dense = true;
    pcl_cloud.header.frame_id = frame_id;
    int i=0;
    for(int r=0;r<height;r++)
    { 
        for(int c=0;c<width;c++)
        {   
            if(depth_image.at<unsigned short>(r,c)==0||depth_image.at<unsigned short>(r,c)!=depth_image.at<unsigned short>(r,c)) continue;
            pcl::PointXYZRGB point;
            point.r = color_image.at<Vec3b>(r,c)[2];
            point.g = color_image.at<Vec3b>(r,c)[1];
            point.b = color_image.at<Vec3b>(r,c)[0];
            point.z = depth_image.at<unsigned short>(r,c) * 0.001;
            point.x = point.z * ( (double)c - cx ) / (fx);
            point.y = point.z * ((double)r - cy ) / (fy);
            pcl_cloud.points.push_back(point);
            i++;
        }
    }
    pcl_cloud.width=i;
    pcl_cloud.resize(i);
    return pcl_cloud;
}

void PCLUtilities::visualizeMesh(const pcl::PolygonMesh& triangles)
{
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPolygonMesh(triangles,"meshes",0);
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "meshes");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    //viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,0.7, 0.7, 0,"meshes"); 
    viewer->setRepresentationToWireframeForAllActors(); 
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
    }
}

pcl::PolygonMesh PCLUtilities::fastMeshGeneration(pcl::PointCloud<PointXYZ>::Ptr cloud)
{
    pcl::PolygonMesh triangles;
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (cloud);
    n.setInputCloud (cloud);
    n.setSearchMethod (tree);
    n.setKSearch (20);
    n.compute (*normals);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields (*cloud, *normals, *cloud_with_normals);
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree2 (new pcl::search::KdTree<pcl::PointNormal>);
    tree2->setInputCloud (cloud_with_normals);
    pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
    gp3.setSearchRadius (0.025);
    gp3.setMu (2.5);
    gp3.setMaximumNearestNeighbors (100);
    gp3.setMaximumSurfaceAngle(M_PI/4);
    gp3.setMinimumAngle(M_PI/18);
    gp3.setMaximumAngle(2*M_PI/3);
    gp3.setNormalConsistency(false);
    gp3.setInputCloud (cloud_with_normals);
    gp3.setSearchMethod (tree2);
    gp3.reconstruct (triangles);
    return triangles;
}
