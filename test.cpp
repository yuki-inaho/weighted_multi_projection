//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> inline unsigned int
pcl::computeCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
                              const pcl::PointIndices &indices,
                              Eigen::Matrix<Scalar, 3, 3> &covariance_matrix)
{
  return (computeCovarianceMatrix (cloud, indices.indices, covariance_matrix));
}

//////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT, typename Scalar> inline unsigned int
pcl::computeMeanAndCovarianceMatrix (const pcl::PointCloud<PointT> &cloud,
                                     Eigen::Matrix<Scalar, 3, 3> &covariance_matrix,
                                     Eigen::Matrix<Scalar, 4, 1> &centroid)
{
  // create the buffer on the stack which is much faster than using cloud[indices[i]] and centroid as a buffer
  Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor> accu = Eigen::Matrix<Scalar, 1, 9, Eigen::RowMajor>::Zero ();
  size_t point_count;
  if (cloud.is_dense)
  {
    point_count = cloud.size ();
    // For each point in the cloud
    for (size_t i = 0; i < point_count; ++i)
    {
      accu [0] += cloud[i].x * cloud[i].x;
      accu [1] += cloud[i].x * cloud[i].y;
      accu [2] += cloud[i].x * cloud[i].z;
      accu [3] += cloud[i].y * cloud[i].y; // 4
      accu [4] += cloud[i].y * cloud[i].z; // 5
      accu [5] += cloud[i].z * cloud[i].z; // 8
      accu [6] += cloud[i].x;
      accu [7] += cloud[i].y;
      accu [8] += cloud[i].z;
    }
  }
  else
  {
    point_count = 0;
    for (size_t i = 0; i < cloud.size (); ++i)
    {
      if (!isFinite (cloud[i]))
        continue;

      accu [0] += cloud[i].x * cloud[i].x;
      accu [1] += cloud[i].x * cloud[i].y;
      accu [2] += cloud[i].x * cloud[i].z;
      accu [3] += cloud[i].y * cloud[i].y;
      accu [4] += cloud[i].y * cloud[i].z;
      accu [5] += cloud[i].z * cloud[i].z;
      accu [6] += cloud[i].x;
      accu [7] += cloud[i].y;
      accu [8] += cloud[i].z;
      ++point_count;
    }
  }
  accu /= static_cast<Scalar> (point_count);
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
