#include"random"
class CNN_layer
{
public:
    float ****out;
    float ****dx;
    float ****dw;
    float *db;
    float ****w;
    float *b;
    float ****W_s;
    float ****W_r;
    float *B_s;
    float *B_r;
    int batch, input_channel, output_channel, input_x,input_y, ksize_x, ksize_y, t;
    // 默认零填充(不可选), 默认步长为[1,1](可选)
    void init(int input_batch, int input_m, int input_n, int input_c, int out_c, int ksize_m, int ksize_n, int strides_x, int strides_y);
    ~CNN_layer();
    void forward(float ****input);
    void backward(float ****input, float ****for_input);
    void optim(float lr, float p1, float p2);
};

class Fully_layer
{
public:
    float **out;
    float **dx;
    float **w;
    float **dw;
    float **w_s;
    float **w_r;
    float *b;
    float *db;
    float *b_s;
    float *b_r;
    int batch, input_channel,output_channel, t;
    void init(int input_batch, int input_c, int output_c);
    ~Fully_layer();
    void forward(float **input);
    void backward(float **input, float **for_input);
    void optim(float lr, float p1, float p2);
};

class ReLU_4D
{
public:
   float ****out;
   float ****dx;
   float ****d_save; // save dout/dx, avoid mismakes when two times across.
   int batch, x, y, c;
   void init(int b, int m, int n, int ch);
   ~ReLU_4D();
   void forward(float ****input);
   void backward(float ****input);
};

class ReLU_2D
{
public:
   float **out;
   float **dx;
   float **d_save;
   int batch, c;
   void init(int b, int ch);
   ~ReLU_2D();
   void forward(float **input);
   void backward(float **input);
};

class MaxPooling
{
public:
   float ****out;
   float ****dx;
   int *local_x;
   int *local_y;
   int batch, input_x, input_y, intput_channel, ksize_x, ksize_y, strides_x, strides_y;
   int output_x, output_y;
   void init(int b, int input_m, int input_n, int c, int ksize_m, int ksize_n, int strides_m, int strides_n);
   ~MaxPooling();
   void forward(float ****input);
   void backward(float ****input);
};

class DropOut
{
public:
   float **dx;
   float **out;
   float **d_save;
   float k_pre;
   int batch, channel;
   void init(int b, int c, float pre);
   ~DropOut();
   void forward(float **input);
   void backward(float **input);
};

class Reshape
{
public:
   float ****dx;
   float **out;
   int batch, input_x, input_y, input_c, output_c;
   void init(int b, int input_m, int input_n, int input_ch);
   ~Reshape();
   void forward(float ****input);
   void backward(float **input);
};

class cat
{
public:
   float **out;
   float **dx1;
   float **dx2;
   int batch, input1_c, input2_c, output_c;
   void init(int b, int input1_ch, int input2_ch);
   ~cat();
   void forward(float **input1, float **input2);
   void backward(float **input);
};

class softmax
{
public:
   float **out;
   float **dx;
   int batch, input_channel;
   void init(int b, int inpuch_c);
   ~softmax();
   void forward(float **input);
   void backward(float **input);
};

class loss
{
public:
   float Loss;
   float **dx;
   int batch, input_channel;
   void init(int b, int input_c);
   ~loss();
   void forward(float **y_pre, int *y_true);
   void backward(float **y_pre, int *y_true);
};

float accurate(float **y_pre, int *y_true, int batch, int classes);