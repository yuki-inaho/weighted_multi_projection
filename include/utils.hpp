#pragma once

#include "header.h"
#include <boost/thread/thread.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/pcl_search.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/extract_indices.h>
#include <iomanip>
#include <fstream>

using namespace std;
using namespace pcl;

double D_MAX = std::numeric_limits<double>::max();
double D_MIN = std::numeric_limits<double>::min();

//////////////////////////////////////////////////////////////////////////////////////////////
inline unsigned int
computeMeanAndWeightedCovarianceMatrix (const pcl::PointCloud<pcl::PointXYZ> &cloud,
                                     const pcl::PointXYZ center_point,
                                     Eigen::Matrix<float, 3, 3> &covariance_matrix,
                                     Eigen::Matrix<float, 4, 1> &centroid,
                                     float sigma)
{
    // create the buffer on the stack which is much faster than using cloud[indices[i]] and centroid as a buffer
    Eigen::Matrix<float, 1, 9, Eigen::RowMajor> accu = Eigen::Matrix<float, 1, 9, Eigen::RowMajor>::Zero ();
    size_t point_count;

    std::vector<float> similarity_vec;
    for(int i=0; i<cloud.points.size(); i++){
        if(std::isfinite(cloud.points[i].x)){
            float sqr_diff_x, sqr_diff_y, sqr_diff_z;
            sqr_diff_x = (cloud.points[i].x - center_point.x) * (cloud.points[i].x - center_point.x);
            sqr_diff_y = (cloud.points[i].y - center_point.y) * (cloud.points[i].y - center_point.y);
            sqr_diff_z = (cloud.points[i].z - center_point.z) * (cloud.points[i].z - center_point.z);
            float diff = sqr_diff_x + sqr_diff_y + sqr_diff_z;
            similarity_vec.push_back(exp(- diff/(2*sigma*sigma))); 
        }else{
            similarity_vec.push_back(0);
        }
    }

    float similarity_sum = 0;
    for(int j=0;j<similarity_vec.size();j++){
        similarity_sum += similarity_vec[j];
    }

    for(int j=0;j<similarity_vec.size();j++){
        similarity_vec[j] /= similarity_sum;
    }

    if (cloud.is_dense)
    {
    point_count = cloud.points.size ();
    // For each point in the cloud
    for (size_t i = 0; i < point_count; ++i)
    {
        accu [0] += cloud.points[i].x * cloud.points[i].x * similarity_vec[i];
        accu [1] += cloud.points[i].x * cloud.points[i].y * similarity_vec[i];
        accu [2] += cloud.points[i].x * cloud.points[i].z * similarity_vec[i];
        accu [3] += cloud.points[i].y * cloud.points[i].y * similarity_vec[i]; // 4
        accu [4] += cloud.points[i].y * cloud.points[i].z * similarity_vec[i]; // 5
        accu [5] += cloud.points[i].z * cloud.points[i].z * similarity_vec[i]; // 8
        accu [6] += cloud.points[i].x * similarity_vec[i];
        accu [7] += cloud.points[i].y * similarity_vec[i];
        accu [8] += cloud.points[i].z * similarity_vec[i];
    }
    }
    else
    {
    point_count = 0;
    for (size_t i = 0; i < cloud.points.size (); ++i)
    {
        if (!isFinite (cloud.points[i]))
        continue;

        accu [0] += cloud.points[i].x * cloud.points[i].x * similarity_vec[i];
        accu [1] += cloud.points[i].x * cloud.points[i].y * similarity_vec[i];
        accu [2] += cloud.points[i].x * cloud.points[i].z * similarity_vec[i];
        accu [3] += cloud.points[i].y * cloud.points[i].y * similarity_vec[i];
        accu [4] += cloud.points[i].y * cloud.points[i].z * similarity_vec[i];
        accu [5] += cloud.points[i].z * cloud.points[i].z * similarity_vec[i];
        accu [6] += cloud.points[i].x * similarity_vec[i];
        accu [7] += cloud.points[i].y * similarity_vec[i];
        accu [8] += cloud.points[i].z * similarity_vec[i];
        ++point_count;
    }
    }
    //accu /= static_cast<float> (point_count);
    if (point_count != 0)
    {
        //centroid.head<3> () = accu.tail<3> ();    -- does not compile with Clang 3.0
        centroid[0] = accu[6]; centroid[1] = accu[7]; centroid[2] = accu[8];
        centroid[3] = 1;
        covariance_matrix.coeffRef (0) = accu [0] - accu [6] * accu [6];
        covariance_matrix.coeffRef (1) = accu [1] - accu [6] * accu [7];
        covariance_matrix.coeffRef (2) = accu [2] - accu [6] * accu [8];
        covariance_matrix.coeffRef (4) = accu [3] - accu [7] * accu [7];
        covariance_matrix.coeffRef (5) = accu [4] - accu [7] * accu [8];
        covariance_matrix.coeffRef (8) = accu [5] - accu [8] * accu [8];
        covariance_matrix.coeffRef (3) = covariance_matrix.coeff (1);
        covariance_matrix.coeffRef (6) = covariance_matrix.coeff (2);
        covariance_matrix.coeffRef (7) = covariance_matrix.coeff (5);
    }
    return (static_cast<unsigned int> (point_count));
}

void 
calculatePointCloudLocalPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, arma::mat &_output, int _K, float _radius, float sigma){
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (cloud);

    int K = _K;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    float radius = _radius;
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    int threads_ = 8;
    arma::mat output;

    output.zeros(cloud->points.size(), 6);
    ////pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);    
    std::chrono::system_clock::time_point  start, end; // 型は auto で可
    start = std::chrono::system_clock::now(); // 計測開始時間

    #pragma omp parallel for shared (output) private (pointIdxNKNSearch, pointNKNSquaredDistance) num_threads(threads_)
    //#pragma omp parallel for shared (output) private (pointIdxRadiusSearch, pointRadiusSquaredDistance) num_threads(threads_)
    for (int idx = 0; idx < static_cast<int> (cloud->points.size ()); ++idx)
    {
        //if ( kdtree.radiusSearch (cloud->points[idx], radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
        if ( kdtree.nearestKSearch (cloud->points[idx], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {
            
            //if(pointIdxRadiusSearch.size()<5) continue;
            pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud (new pcl::PointCloud<pcl::PointXYZ>);

            //for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i){                
            for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i){
                pcl::PointXYZ point;
                
                point.x = cloud->points[pointIdxNKNSearch[i]].x;
                point.y = cloud->points[pointIdxNKNSearch[i]].y;
                point.z = cloud->points[pointIdxNKNSearch[i]].z;

                /*
                point.x = cloud->points[pointIdxRadiusSearch[i]].x;
                point.y = cloud->points[pointIdxRadiusSearch[i]].y;
                point.z = cloud->points[pointIdxRadiusSearch[i]].z;
                */

                _cloud->points.push_back(point);
            }

            EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
            EIGEN_ALIGN16 Eigen::Vector3f eigen_value;
            EIGEN_ALIGN16 Eigen::Matrix3f eigen_vector;
            Eigen::Vector4f xyz_centroid;
            
            computeMeanAndWeightedCovarianceMatrix (*_cloud, cloud->points[idx], covariance_matrix, xyz_centroid, sigma);
            //pcl::eigen33 (covariance_matrix, eigen_vector, eigen_value);

            pcl::eigen33 (covariance_matrix, eigen_vector, eigen_value); // ????

            float a_x = eigen_vector(0);
            float a_y = eigen_vector(1);
            float a_z = eigen_vector(2);
            // Compute the curvature surface change
            float c = a_x * xyz_centroid[0] + a_y * xyz_centroid[1] + a_z * xyz_centroid[2];

            //to arrange eigen values to ascent order 
            output(idx, 3) = double(c);
            output(idx, 2) = double(a_z);
            output(idx, 1) = double(a_y);
            output(idx, 0) = double(a_x);
        }
    }
    _output = output;

    end = std::chrono::system_clock::now();  // 計測終了時間
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間をミリ秒に変換

    cout <<"num cloud" << cloud->points.size() << endl;
    std::cout << output.n_rows << std::endl;
    std::cout << "elapsed:"<< elapsed << std::endl;
}

void
LocalPlane2RGB(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const arma::mat &primitiv, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb){
    int pc_size = cloud->points.size();

    cloud_rgb->points.clear();

    for(int i=0;i<pc_size;i++){
        pcl::PointXYZRGB _point_rgb;
        _point_rgb.x = cloud->points[i].x;
        _point_rgb.y = cloud->points[i].y;
        _point_rgb.z = cloud->points[i].z;

        //thinny
        _point_rgb.r = static_cast<unsigned char>(primitiv(i,0) * 255.0);
        _point_rgb.g = static_cast<unsigned char>(primitiv(i,1) * 255.0);
        _point_rgb.b = static_cast<unsigned char>(primitiv(i,2) * 255.0);

        cloud_rgb->points.push_back(_point_rgb);
    }
}

//removeNan: NaN要素を点群データから除去するメソッド
//input : target(NaN要素を除去する対象の点群)
//output: cloud(除去を行った点群)
pcl::PointCloud<PointXYZ>::Ptr removeNan(pcl::PointCloud<pcl::PointXYZ>::Ptr target){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  int n_point = target->points.size();
  for(int i=0;i<n_point; i++){
    pcl::PointXYZ tmp_point;
    if(std::isfinite(target->points[i].x) || std::isfinite(target->points[i].y) || std::isfinite(target->points[i].z)){
      tmp_point.x = target->points[i].x;
      tmp_point.y = target->points[i].y;
      tmp_point.z = target->points[i].z;
      cloud->points.push_back(tmp_point);
    }
  }
//  cout << "varid points:" << cloud->points.size() << endl;
  return cloud;


}

void
getTPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,  arma::mat &plane_param, int point_idx, int plane_idx, pcl::PointXYZ &point_T){
    float a_x = plane_param(plane_idx,0);
    float a_y = plane_param(plane_idx,1);
    float a_z = plane_param(plane_idx,2);
    float c = plane_param(plane_idx,3);
    float aj =  a_x * cloud->points[point_idx].x + a_y * cloud->points[point_idx].y + a_z * cloud->points[point_idx].z;
    point_T.x = cloud->points[point_idx].x - aj * a_x + c* a_x ;
    point_T.y = cloud->points[point_idx].y - aj * a_y + c* a_y ;
    point_T.z = cloud->points[point_idx].z - aj * a_z + c* a_z ;
}

void
denoiseWMP(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,  arma::mat &plane_param, int _K, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_denoised, float sigma){
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (cloud);

    int K = _K;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);

    int threads_ = 8;

//    #pragma omp parallel for shared (cloud_denoised) private (pointIdxNKNSearch, pointNKNSquaredDistance) num_threads(threads_)
    for (int idx = 0; idx < static_cast<int> (cloud->points.size ()); ++idx)
    {
        if ( kdtree.nearestKSearch (cloud->points[idx], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {


            pcl::PointXYZ center_point = cloud->points[idx];

            std::vector<float> similarity_vec;
            for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i){
                float sqr_diff_x, sqr_diff_y, sqr_diff_z;
                sqr_diff_x = (cloud->points[pointIdxNKNSearch[i]].x - center_point.x) * (cloud->points[pointIdxNKNSearch[i]].x - center_point.x);
                sqr_diff_y = (cloud->points[pointIdxNKNSearch[i]].y - center_point.y) * (cloud->points[pointIdxNKNSearch[i]].y - center_point.y);
                sqr_diff_z = (cloud->points[pointIdxNKNSearch[i]].z - center_point.z) * (cloud->points[pointIdxNKNSearch[i]].z - center_point.z);
                float diff = sqr_diff_x + sqr_diff_y + sqr_diff_z;
                float similarity = exp(- diff/(2*sigma*sigma)); 
                similarity_vec.push_back(similarity);
            }

            float similarity_sum = 0;
            for(int j=0;j<similarity_vec.size();j++){
                similarity_sum += similarity_vec[j];
            }

            for(int j=0;j<similarity_vec.size();j++){
                similarity_vec[j] /= similarity_sum;
            }
            //pcl::PointXYZ point;
            //getTPoints(cloud, plane_param, idx ,pointIdxNKNSearch[i], point);
            //getTPoints(cloud, plane_param, idx , idx, point);

            pcl::PointXYZ denoised_point;
            denoised_point.x = 0; denoised_point.y = 0; denoised_point.z = 0;
            for (size_t i = 0; i < pointIdxNKNSearch.size (); ++i){
                pcl::PointXYZ point;
                getTPoints(cloud, plane_param, idx ,pointIdxNKNSearch[i], point);
                denoised_point.x += similarity_vec[i] * point.x;
                denoised_point.y += similarity_vec[i] * point.y;
                denoised_point.z += similarity_vec[i] * point.z;
            }

            cloud_denoised->points.push_back(denoised_point);
            //cloud_denoised->points.push_back(point);
        }
    }
    cout << cloud_denoised->points.size() << endl;
}


pcl::visualization::PCLVisualizer::Ptr simpleVis ()
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}
