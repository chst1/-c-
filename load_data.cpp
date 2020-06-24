#include"load_data.h"
#include"utils_other.h"
#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;

load_data::load_data(string file_name, int Batch, int image_x, int image_y, int image_z)
{
    batch = Batch;
    size_x = image_x;
    size_y = image_y;
    size_z = image_z;
    fstream write_file(file_name, ofstream::in);
    number = 0;
    now_num = 0;
    string path,lab;
    while(write_file>>path>>lab)
    {
        image_file.push_back(path);
        int n = 0;
        for(int i=0;i<lab.size();i++)
        {
            n = n * 10 + int(lab[i]-'0');
        }
        label.push_back(n);
        number++;
    }
    for(int i=0;i<number;i++)
    {
        rand_index.push_back(i);
    }
    // rand
    random_shuffle(rand_index.begin(), rand_index.end());
}

void load_data::next_batch(float ****image, int *y_true)
{
    if(now_num+batch>number)
    {
        now_num = 0;
        random_shuffle(rand_index.begin(), rand_index.end());
    }
    for(int i=now_num;i<now_num+batch;i++)
    {
        Mat mat;
        Mat dst;
        mat = imread(image_file[rand_index[i]], CV_LOAD_IMAGE_COLOR);
        resize(mat, dst, Size(size_x, size_y));
        // namedWindow("Display",  CV_WINDOW_AUTOSIZE);
        // imshow("Display", mat);
        // waitKey(1000);
        for(int x=0;x<size_x;x++)
        {
            for(int y=0;y<size_y;y++)
            {
                for(int j=0;j<3;j++)
                {
                    //cout<<int(mat.ptr<Vec3b>(x)[y][j])<<" "<<endl;
                    // cout<<mat.at<Vec3b>(x,y)[j]<<" "<<endl;
                    image[i-now_num][x][y][j] = int(dst.at<Vec3b>(x,y)[j])*1.0/255.-0.5;
                }
            }
        }
        y_true[i-now_num] = label[rand_index[i]];
    }
    now_num += batch;
}