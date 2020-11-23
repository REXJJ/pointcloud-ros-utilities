#ifndef PCL_UTILITIES_HPP
#define PCL_UTILITIES_HPP

/*******************************************/
//ROS HEADERS
/********************************************/
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

/*********************************************/
//PCL HEADERS
/**********************************************/
#include <pcl/conversions.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>

/***********************************************/
//STANDARD HEADERS
/************************************************/
#include <iostream>
#include <cmath>
#include <vector>
#include <utility>
#include <chrono>
#include <unordered_map> 
#include <queue>
#include <fstream>
#include <thread>
#include <ctime>

#include <cv.h>
#include <highgui.h>

#include <Eigen/Dense>

using namespace std;
using namespace pcl;
using namespace Eigen;
using namespace cv;


class PCLUtilities
{  
    public:
        static vector<int> splitRGBData(float rgb);
        template <typename PointT> static vector<double> pointsToVector(PointT t);
        /***************************************************************************/
        //Point Cloud to other Data Structures
        /***************************************************************************/
        template <typename PointT> static vector<vector<double> > pclToVector(const pcl::PointCloud<PointT>& p);
        template <typename PointT> static MatrixXf pclToEigen(const pcl::PointCloud<PointT>& p);
        template <typename PointT> static pcl::PCLPointCloud2 pclToPointCloud2(const pcl::PointCloud<PointT>& p);	
        /***************************************************************************/
        //Other Data Structures to Point Cloud
        /***************************************************************************/
        static vector<vector<float> > pointCloud2ToVec(const pcl::PCLPointCloud2& p);
        static pcl::PointCloud<pcl::PointXYZ> pointCloud2ToPclXYZ(const pcl::PCLPointCloud2& p);
        static pcl::PointCloud<pcl::PointXYZRGB> pointCloud2ToPclXYZRGB(const pcl::PCLPointCloud2& p);
        static vector<vector<float> > pointCloud2ToVec(const sensor_msgs::PointCloud2& p);
        /*****************************************************************************/
        //Point Cloud File Handling Functions
        /****************************************************************************/	
        template <typename PointT> static void pclToCSV(const pcl::PointCloud<PointT>& p, std::string filename);
        template <typename PointT> static void pclToXYZ(const pcl::PointCloud<PointT>& p, std::string filename);
        static void xyzToPcd (const string &input_file, const string &output_file);
        static pcl::PCLPointCloud2 pcdToPointCloud2 (const std::string &filename);
        static void pointCloud2ToPly (const std::string &filename, const pcl::PCLPointCloud2 &cloud, bool format);
        template <typename PointT> static pcl::PointCloud<PointT> PlyToPcl(std::string filename);
        template <typename PointT> static void PclToPcd(std::string filename,const pcl::PointCloud<PointT>& cloud);
        template <typename PointT> static pcl::PointCloud<PointT> PcdToPcl(std::string filename);
        /*******************************************************************/
        //Visualization Utilities
        /******************************************************************/
        template <typename PointT> static void visualizePointCloud(const pcl::PointCloud<PointT>& cloud);
        template <typename PointT> static void visualizePointCloud(const pcl::PointCloud<PointT>& cloud,pcl::visualization::PCLVisualizer::Ptr viewer);
        static void visualizeMesh(const pcl::PolygonMesh& triangles);
        template <typename PointT> static pcl::PointCloud<PointT> downsample(pcl::PointCloud<PointT> cloud,double leaf=0.01);
        template <typename PointT> static pcl::PointCloud<PointT> downsample(typename pcl::PointCloud<PointT>::Ptr cloud, double leaf=0.01);
        static pcl::PointCloud<pcl::PointXYZRGB> makePointCloud(Mat& color_image, Mat& depth_image, Eigen::VectorXd& K, std::string& frame_id);
        template <typename PointT> static void publishPointCloud(const pcl::PointCloud<PointT>& cloud,const ros::Publisher& publish_cloud);
        static pcl::PolygonMesh fastMeshGeneration(pcl::PointCloud<PointXYZ>::Ptr cloud);
};

vector<int> PCLUtilities::splitRGBData(float rgb)
{
    uint32_t data = *reinterpret_cast<int*>(&rgb);
    vector<int> d;
    int a[3]={16,8,1};
    for(int i=0;i<3;i++)
    {
        d.push_back((data>>a[i]) & 0x0000ff);
    }
    return d;
}

template <typename PointT> vector<double> PCLUtilities::pointsToVector(PointT t)
{
    vector<double> d;
    for(int i=0;i<3;i++)
        d.push_back(t.data[i]);
    return d;
}

template <typename PointT> vector<vector<double> > PCLUtilities::pclToVector(const pcl::PointCloud<PointT>& p)
{
    vector<vector<double> > v;
    for(size_t i=0;i<p.points.size();i++)
        v.push_back(pointsToVector(p.points[i]));
    return v;
}

template <typename PointT> MatrixXf PCLUtilities::pclToEigen(const pcl::PointCloud<PointT>& p)
{
    MatrixXf m = p.getMatrixXfMap().transpose();
    if(m.cols()<5) return m;
    for(int i=0;i<m.rows();i++)
    {
        vector<int> colors = splitRGBData(double(m(i,4)));
        m(i,4) = float(colors[0]); 
        m(i,5) = float(colors[1]);
        m(i,6) = float(colors[2]);
    }
    return m;
}

template <typename PointT> pcl::PCLPointCloud2 PCLUtilities::pclToPointCloud2(const pcl::PointCloud<PointT>& p)
{
    pcl::PCLPointCloud2 cloud;
    pcl::toPCLPointCloud2(p,cloud);
    return cloud;
}	


template <typename PointT> pcl::PointCloud<PointT> PCLUtilities::downsample(pcl::PointCloud<PointT> cloud, double leaf)
{
    pcl::VoxelGrid<PointT> sor;
    pcl::PointCloud<PointT> cloud_filtered;
    sor.setLeafSize (leaf, leaf, leaf);
    sor.setInputCloud (cloud.makeShared());
    sor.filter (cloud_filtered);
    return cloud_filtered;
}

template <typename PointT> pcl::PointCloud<PointT> PCLUtilities::downsample(typename pcl::PointCloud<PointT>::Ptr cloud, double leaf)
{
    pcl::VoxelGrid<PointT> sor;
    pcl::PointCloud<PointT> cloud_filtered;
    sor.setLeafSize (leaf, leaf, leaf);
    sor.setInputCloud (cloud);
    sor.filter (cloud_filtered);
    return cloud_filtered;
}

template <typename PointT> void PCLUtilities::pclToCSV(const pcl::PointCloud<PointT>& p, std::string filename)
{
    ofstream f;
    f.open(filename);
    MatrixXf v = pclToEigen<PointT>(p);
    for(int i=0;i<v.rows();i++)
    {
        for(int j=0;j<v.cols()-1;j++)
            f<<v(i,j)<<",";
        f<<v(i,v.cols()-1)<<"\n";
    }
    f.close();
}

template <typename PointT> void PCLUtilities::pclToXYZ(const pcl::PointCloud<PointT>& p, std::string filename)
{
    ofstream f;
    f.open(filename);
    MatrixXf v = pclToEigen<PointT>(p);
    for(int i=0;i<v.rows();i++)
    {
        for(int j=0;j<2;j++)
            f<<v(i,j)<<" ";
        f<<v(i,2)<<"\n";
    }
    f.close();
}

void PCLUtilities::pointCloud2ToPly(const std::string &filename, const pcl::PCLPointCloud2 &cloud, bool format)
{
    std::cout<<"Saving \n"; printf ("%s ", filename.c_str ());
    pcl::PCDWriter writer;
    writer.write (filename, cloud, Eigen::Vector4f::Zero (), Eigen::Quaternionf::Identity (), format);
}

template <typename PointT> pcl::PointCloud<PointT> PCLUtilities::PlyToPcl(std::string filename)
{
    pcl::PointCloud<PointT> cloud;
    pcl::PLYReader Reader;
    Reader.read(filename, cloud);
}

template <typename PointT> void PCLUtilities::PclToPcd(std::string filename,const pcl::PointCloud<PointT>& cloud)
{
    pcl::io::savePCDFileASCII (filename, cloud);
}


template <typename PointT> pcl::PointCloud<PointT> PCLUtilities::PcdToPcl(std::string filename)
{
    pcl::PointCloud<PointT> cloud;
    if (pcl::io::loadPCDFile<PointT> (filename,cloud) == -1)
    {
        PCL_ERROR ("Couldn't read the inputted file. \n");
    }
    return cloud;
}

template <typename PointT> inline void PCLUtilities::publishPointCloud(const pcl::PointCloud<PointT>& cloud,const ros::Publisher& publish_cloud)
{
    pcl::PCLPointCloud2* cloud_output = new pcl::PCLPointCloud2; 
    pcl::PCLPointCloud2ConstPtr cloudPtr(cloud_output);
    pcl::toPCLPointCloud2(cloud, *cloud_output);
    sensor_msgs::PointCloud2 output;
    pcl_conversions::fromPCL(*cloud_output, output);   
    output.is_bigendian = false;
    output.header.seq=1;
    output.header.stamp=ros::Time::now();
    output.header.frame_id=cloud.header.frame_id;
    output.height = cloud.height;
    output.width = cloud.width; 
    publish_cloud.publish (output);
} 


/*******************************************************************/
//Visualization Utilities
/******************************************************************/

template <typename PointT> void PCLUtilities::visualizePointCloud(const pcl::PointCloud<PointT>& cloud)
{
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<PointT> (cloud.makeShared(), "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);

    }
}

template <typename PointT> void PCLUtilities::visualizePointCloud(const pcl::PointCloud<PointT>& cloud,pcl::visualization::PCLVisualizer::Ptr viewer)
{
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<PointT> (cloud.makeShared(), "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
}
#endif
