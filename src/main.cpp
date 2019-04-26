#include "header.h"
#include "utils.hpp"
#include "ParameterManager.hpp"
#include <pcl/filters/statistical_outlier_removal.h>

using namespace pcl;

std::string CFG_PARAM_PATH = "/home/inaho-00/work/cpp/weighted_multi_projection/cfg/recognition_parameter.toml";

int
main (int argc, char** argv)
{
    ParameterManager cfg_param(CFG_PARAM_PATH);
    float sigma = cfg_param.ReadFloatData("Param", "sigma");
    float param_R = cfg_param.ReadFloatData("Param", "param_R");
    int param_K = cfg_param.ReadIntData("Param", "param_K");
    std::string DATA_PATH = cfg_param.ReadStringData("Param", "data_path");

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ> (DATA_PATH, *cloud) == -1) 
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }

    cloud = removeNan(cloud);
    cout << cloud->points.size()  << endl;

    // Create the filtering object
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud (cloud);
    sor.setMeanK (10);
    sor.setStddevMulThresh (0.5);
    sor.filter (*cloud);

    arma::mat output;
    /*
    pcl::VoxelGrid<pcl::PointXYZ> vor;
    vor.setInputCloud (cloud);
    vor.setLeafSize (0.003f, 0.003f, 0.003f);
    vor.filter (*cloud);    
    */
    cout << cloud->points.size()  << endl;

    calculatePointCloudLocalPlane(cloud, output, param_K, param_R, sigma);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_denoised (new pcl::PointCloud<pcl::PointXYZ>);
    denoiseWMP(cloud, output, param_K, cloud_denoised, sigma);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb (new pcl::PointCloud<pcl::PointXYZRGB>);
    //LocalPlane2RGB(cloud, output, cloud_rgb);

    pcl::visualization::PCLVisualizer::Ptr viewer;
    viewer = simpleVis();
    viewer->addPointCloud<pcl::PointXYZ> (cloud_denoised, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud");  

    pcl::io::savePCDFileBinary("../data/denoised.pcd", *cloud_denoised);

    //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_rgb);
    //viewer->addPointCloud<pcl::PointXYZRGB> (cloud_rgb, rgb, "sample cloud");
    //viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud");  

    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        boost::this_thread::sleep (boost::posix_time::microseconds (100000));
    }

    return (0);
}