#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <chrono>
#include <dirent.h>

void eucli_cluster(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                   std::unordered_map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> &clusters,
                   float step, int min_points, int max_points)
{
    if (cloud->points.size() < 2)
        return;

    //rasterized
    float min_y = HUGE_VALF;
    float min_z = HUGE_VALF;
    float max_y = -HUGE_VALF;
    float max_z = -HUGE_VALF;

    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        pcl::PointXYZ &p_tmp = cloud->points[i];
        min_y = std::min(min_y, p_tmp.y);
        min_z = std::min(min_z, p_tmp.z);
        max_y = std::max(max_y, p_tmp.y);
        max_z = std::max(max_z, p_tmp.z);
    }

    size_t cols = static_cast<size_t>(std::ceil((max_y - min_y) / step));
    size_t rows = static_cast<size_t>(std::ceil((max_z - min_z) / step));

    std::vector<std::vector<int>> raster(rows, std::vector<int>(cols, 0));
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        int c_idx = static_cast<int>((cloud->points[i].y - min_y) / step);
        int r_idx = static_cast<int>((cloud->points[i].z - min_z) / step);
        raster[r_idx][c_idx] = 1;
    }

    //get connect compoent，use two pass
    std::unordered_map<int, int> label_map;
    int label = 1;
    //pass 1
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            if (raster[i][j] > 0)
            {
                int top = i > 0 ? raster[i - 1][j] : 0;
                int left = j > 0 ? raster[i][j - 1] : 0;

                if (top == 0 && left == 0)
                {
                    raster[i][j] = label;
                    label_map.insert(std::make_pair(label, label));
                    label++;
                }
                else if (top > 0 && left > 0)
                {
                    raster[i][j] = std::min(top, left);
                    int &val_top = label_map[top];
                    int &val_left = label_map[left];
                    int label_min = std::min(val_top, val_left);
                    val_top = label_min;
                    val_left = label_min;
                }
                else if (top > 0 || left > 0)
                {
                    raster[i][j] = top > 0 ? top : left;
                }
            }
        }
    }

    //pass 2
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            if (raster[i][j] > 0)
            {
                int label_min = label_map[raster[i][j]];
                raster[i][j] = label_min;
            }
        }
    }

    //get clusters
    for (size_t i = 0; i < cloud->points.size(); ++i)
    {
        int c_idx = static_cast<int>((cloud->points[i].y - min_y) / step);
        int r_idx = static_cast<int>((cloud->points[i].z - min_z) / step);

        auto it = clusters.find(raster[r_idx][c_idx]);
        if (it != clusters.end())
        {
            it->second->points.emplace_back(cloud->points[i]);
            it->second->width++;
        }
        else
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            cluster_cloud->points.emplace_back(cloud->points[i]);
            cluster_cloud->height = 1;
            cluster_cloud->width = 1;
            clusters.insert(std::make_pair(raster[r_idx][c_idx], cluster_cloud));
        }
    }

    //optimization
    for (auto it = clusters.begin(); it != clusters.end();)
    {
        int sz = static_cast<int>(it->second->points.size());
        if (sz > max_points || sz < min_points)
            clusters.erase(it++);
        else
            it++;
    }
}

using namespace std;

void CreateCloudFromTxt(const std::string &file_path, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    std::ifstream file(file_path.c_str()); //c_str()：生成一个const char*指针，指向以空字符终止的数组。
    std::string line;
    pcl::PointXYZ point;
    while (getline(file, line))
    {
        std::stringstream ss(line);
        std::string tmp;
        vector<double> str_list;
        while (getline(ss, tmp, ','))
        {
            str_list.emplace_back(std::stod(tmp));
        }

        cloud->points.emplace_back(pcl::PointXYZ(str_list[0], str_list[1], str_list[2]));
    }
    file.close();
}

bool get_filelist_from_dir(const std::string &_path, std::vector<std::string> &_files)
{
    DIR *dir;
    dir = opendir(_path.c_str());
    struct dirent *ptr;
    std::vector<std::string> file;
    while ((ptr = readdir(dir)) != NULL)
    {
        if (ptr->d_name[0] == '.')
        {
            continue;
        }
        file.push_back(ptr->d_name);
    }
    closedir(dir);
    sort(file.begin(), file.end());
    _files = file;
}

using namespace std::chrono;
int main()
{
    ofstream ofiles("./time_spend.txt", ios::app);
    vector<std::string> folders;
    get_filelist_from_dir("/home/jin/Downloads/saved_pointclouds/", folders);

    for (int f = 10; f < folders.size(); ++f)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        string data_path = "/home/jin/Downloads/saved_pointclouds/" + folders[f] + "/";
        vector<string> data_list_path;
        get_filelist_from_dir(data_path, data_list_path);
        for (int i = 0; i < data_list_path.size(); ++i)
        {
            if (data_list_path[i].find("bbox_infos") == data_list_path[i].npos)
            {
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);
                CreateCloudFromTxt(data_path + data_list_path[i], cloud_tmp);
                *cloud += *cloud_tmp;
            }
        }
        
        std::unordered_map<int, pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
        high_resolution_clock::time_point sync_process_start2 = high_resolution_clock::now();
        //FindCluster(cloud, out_clusters);
        eucli_cluster(cloud, clusters, 0.5, 20, 5000);

        auto sync_process_end2 = high_resolution_clock::now();
        auto events_time_interval2 = duration_cast<duration<double>>(sync_process_end2 - sync_process_start2).count();
        std::cout << "[Total spend CPU] events_time_interval: " << events_time_interval2 << "\n";

        ofiles << "points:" << cloud->points.size() << " clusters:" << clusters.size() << " CPU time interval:" << events_time_interval2 << endl;
        /*
        for (auto it = clusters.begin(); it != clusters.end(); ++it)
        {
            string path = "./clusters/" + to_string(it->first) + ".txt";
            ofstream ofiles2(path, ios::app);
            for (int j = 0; j < it->second->points.size(); ++j)
            {
                ofiles2 << it->second->points[j].x << "," << it->second->points[j].y << "," << it->second->points[j].z << endl;
            }
        }

        break;
        */
    }

    return 0;
}