#include<fstream>
#include<opencv2/core/mat.hpp>
#include<string>
#include<vector>
using namespace std;

class load_data
{
    int batch, size_x, size_y, size_z, number, now_num;
    vector<string> image_file;
    vector<int>rand_index;
    vector<int> label;
public:
    load_data(string file_name, int Batch, int image_x, int image_y, int image_z);
    void next_batch(float ****image, int *y_true);
};