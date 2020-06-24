#include "random"
#include "layer.h"
#include "utils.h"
#include"utils_other.h"
void CNN_layer::init(int input_batch, int input_m, int input_n, int input_c, int out_c, int ksize_m, int ksize_n, int strides_x = 1, int strides_y = 1)
{
    t = 0;
    batch = input_batch;
    input_channel = input_c;
    output_channel = out_c;
    input_x = input_m;
    input_y = input_n;
    ksize_x = ksize_m;
    ksize_y = ksize_n;
    w = get_mem<float>(ksize_x, ksize_y, input_channel, output_channel);
    W_s = get_mem<float>(ksize_x, ksize_y, input_channel, output_channel);
    W_r = get_mem<float>(ksize_x, ksize_y, input_channel, output_channel);
    dw = get_mem<float>(ksize_x, ksize_y, input_channel, output_channel);
    out = get_mem<float>(batch, input_x, input_y, output_channel);
    dx = get_mem<float>(batch, input_x, input_y, input_channel);
    b = new float[output_channel];
    db = new float[output_channel];
    B_s = new float[output_channel];
    B_r = new float[output_channel];
    for (int i = 0; i < output_channel; i++)
    {
        db[i] = 0;
        B_r[i] = 0;
        B_s[i] = 0;
    }
    init_w(w, ksize_x, ksize_y, input_channel, output_channel, MEAN, VARIANCE);
    init_b(b, output_channel, MEAN, VARIANCE);
}

void CNN_layer::save_arg(fstream& file)
{
    for(int i = 0; i<ksize_x; i++)
    {
        for(int j = 0; j<ksize_y; j++)
        {
            for(int p=0;p<input_channel; p++)
            {
                for(int q =0; q<output_channel; q++)
                {
                    file<<w[i][j][p][q]<<" ";
                }
            }
        }
    }
    for(int i=0;i<output_channel;i++)
    {
        file<<b[i]<<" ";
    }
}

void CNN_layer::load_arg(fstream& file)
{
    for(int i = 0; i<ksize_x; i++)
    {
        for(int j = 0; j<ksize_y; j++)
        {
            for(int p=0;p<input_channel; p++)
            {
                for(int q =0; q<output_channel; q++)
                {
                    file>>w[i][j][p][q];
                }
            }
        }
    }
    for(int i=0;i<output_channel;i++)
    {
        file>>b[i];
    }
}

CNN_layer::~CNN_layer()
{

    delete_mem<float>(out, batch, input_x, input_y, output_channel);
    delete_mem<float>(dx, batch, input_x, input_y, input_channel);
    delete_mem<float>(w, ksize_x, ksize_y, input_channel, output_channel);
    delete_mem<float>(dw, ksize_x, ksize_y, input_channel, output_channel);
    delete_mem<float>(W_r, ksize_x, ksize_y,input_channel, output_channel);
    delete_mem<float>(W_s, ksize_x, ksize_y, input_channel, output_channel);
    delete[] b;
    delete[] db;
    delete[] B_r;
    delete[] B_s;
}

void CNN_layer::forward(float ****input)
{
    int center_x = int(ksize_x / 2), center_y = int(ksize_y / 2);
    for (int i = 0; i < batch; i++)
    {
        for (int x = 0; x < center_x; x++)
        {
            for (int y = 0; y < input_y; y++)
            {
                for (int j = 0; j < output_channel; j++)
                {
                    out[i][x][y][j] = b[j];
                    for (int p = -center_x; p <= center_x; p++)
                    {
                        if (x - p < 0 || x + p >= input_x || x-p>=input_x || x+p<0)
                        {
                            out[i][x][y][j] += 0;
                        }
                        else
                        {
                            for (int q = -center_y; q <= center_y; q++)
                            {
                                if (y - q < 0 || y + q >= input_y || y - q >= input_y || y + q < 0)
                                {
                                    out[i][x][y][j] += 0;
                                }
                                else
                                {
                                    for (int k = 0; k < input_channel; k++)
                                    {
                                        out[i][x][y][j] += w[center_x - p][center_y - q][k][j] * (input[i][x - p][y - q][k]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < batch; i++)
    {
        for (int x = input_x - center_x; x < input_x; x++)
        {
            for (int y = 0; y < input_y; y++)
            {
                for (int j = 0; j < output_channel; j++)
                {
                    out[i][x][y][j] = b[j];
                    for (int p = -center_x; p <= center_x; p++)
                    {
                        if (x - p < 0 || x + p >= input_x || x-p>=input_x || x+p<0)
                        {
                            out[i][x][y][j] += 0;
                        }
                        else
                        {
                            for (int q = -center_y; q <= center_y; q++)
                            {
                                if (y - q < 0 || y + q >= input_y || y - q >= input_y || y + q < 0)
                                {
                                    out[i][x][y][j] += 0;
                                }
                                else
                                {
                                    for (int k = 0; k < input_channel; k++)
                                    {
                                        out[i][x][y][j] += w[center_x - p][center_y - q][k][j] * input[i][x - p][y - q][k];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < batch; i++)
    {
        for (int x = center_x; x < input_x - center_x; x++)
        {
            for (int y = 0; y < center_y; y++)
            {
                for (int j = 0; j < output_channel; j++)
                {
                    out[i][x][y][j] = b[j];
                    for (int p = -center_x; p <= center_x; p++)
                    {
                        if (x - p < 0 || x + p >= input_x || x-p>=input_x || x+p<0)
                        {
                            out[i][x][y][j] += 0;
                        }
                        else
                        {
                            for (int q = -center_y; q <= center_y; q++)
                            {
                                if (y - q < 0 || y + q >= input_y || y - q >= input_y || y + q < 0)
                                {
                                    out[i][x][y][j] += 0;
                                }
                                else
                                {
                                    for (int k = 0; k < input_channel; k++)
                                    {
                                        out[i][x][y][j] += w[center_x - p][center_y - q][k][j] * input[i][x - p][y - q][k];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < batch; i++)
    {
        for (int x = center_x; x < input_x - center_x; x++)
        {
            for (int y = input_y - center_y; y < input_y; y++)
            {
                for (int j = 0; j < output_channel; j++)
                {
                    out[i][x][y][j] = b[j];
                    for (int p = -center_x; p <= center_x; p++)
                    {
                        if (x - p < 0 || x + p >= input_x || x-p>=input_x || x+p<0)
                        {
                            out[i][x][y][j] += 0;
                        }
                        else
                        {
                            for (int q = -center_y; q <= center_y; q++)
                            {
                                if (y - q < 0 || y + q >= input_y || y - q >= input_y || y + q < 0)
                                {
                                    out[i][x][y][j] += 0;
                                }
                                else
                                {
                                    for (int k = 0; k < input_channel; k++)
                                    {
                                        out[i][x][y][j] += w[center_x - p][center_y - q][k][j] * input[i][x - p][y - q][k];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < batch; i++)
    {
        for (int x = center_x; x < input_x - center_x; x++)
        {
            for (int y = center_y; y < input_y - center_y; y++)
            {
                for (int j = 0; j < output_channel; j++)
                {
                    out[i][x][y][j] = b[j];
                    for (int p = -center_x; p <= center_x; p++)
                    {
                        for (int q = -center_y; q <= center_y; q++)
                        {
                            for (int k = 0; k < input_channel; k++)
                            {
                                out[i][x][y][j] += w[center_x - p][center_y - q][k][j] * input[i][x - p][y - q][k];
                            }
                        }
                    }
                }
            }
        }
    }
}

void CNN_layer::backward(float ****input, float ****for_input)
{
    int center_x = int(ksize_x / 2), center_y = int(ksize_y / 2);
    // 求dx
    for (int i = 0; i < batch; i++)
    {
        for (int x = 0; x < center_x; x++)
        {
            for (int y = 0; y < input_y; y++)
            {
                for (int j = 0; j < input_channel; j++)
                {
                    for (int p = -center_x; p <= center_x; p++)
                    {
                        if (x - p < 0 || x + p >= input_y || x-p>=input_y || x+p<0)
                        {
                            dx[i][x][y][j] += 0;
                        }
                        else
                        {
                            for (int q = -center_y; q <= center_y; q++)
                            {
                                if (y - q < 0 || y + q >= input_y || y - q >= input_y || y + q < 0)
                                {
                                    dx[i][x][y][j] += 0;
                                }
                                else
                                {
                                    for (int k = 0; k < output_channel; k++)
                                    {
                                        dx[i][x][y][j] += w[ksize_x - 1 - (center_x - p)][ksize_y - 1 - (center_y - q)][j][k] * input[i][x - p][y - q][k];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < batch; i++)
    {
        for (int x = input_x - center_x; x < input_x; x++)
        {
            for (int y = 0; y < input_y; y++)
            {
                for (int j = 0; j < input_channel; j++)
                {
                    for (int p = -center_x; p <= center_x; p++)
                    {
                        if (x - p < 0 || x + p >= input_y || x-p>=input_y || x+p<0)
                        {
                            dx[i][x][y][j] += 0;
                        }
                        else
                        {
                            for (int q = -center_y; q <= center_y; q++)
                            {
                                if (y - q < 0 || y + q >= input_y || y - q >= input_y || y + q < 0)
                                {
                                    dx[i][x][y][j] += 0;
                                }
                                else
                                {
                                    for (int k = 0; k < output_channel; k++)
                                    {
                                        dx[i][x][y][j] += w[ksize_x - 1 - (center_x - p)][ksize_y - 1 - (center_y - q)][j][k] * input[i][x - p][y - q][k];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < batch; i++)
    {
        for (int x = center_x; x < input_x - center_x; x++)
        {
            for (int y = 0; y < center_y; y++)
            {
                for (int j = 0; j < input_channel; j++)
                {
                    for (int p = -center_x; p <= center_x; p++)
                    {
                        if (x - p < 0 || x + p >= input_y || x-p>=input_y || x+p<0)
                        {
                            dx[i][x][y][j] += 0;
                        }
                        else
                        {
                            for (int q = -center_y; q <= center_y; q++)
                            {
                                if (y - q < 0 || y + q >= input_y || y - q >= input_y || y + q < 0)
                                {
                                    dx[i][x][y][j] += 0;
                                }
                                else
                                {
                                    for (int k = 0; k < output_channel; k++)
                                    {
                                        dx[i][x][y][j] += w[ksize_x - 1 - (center_x - p)][ksize_y - 1 - (center_y - q)][j][k] * input[i][x - p][y - q][k];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < batch; i++)
    {
        for (int x = center_x; x < input_x - center_x; x++)
        {
            for (int y = input_y - center_y; y < input_y; y++)
            {
                for (int j = 0; j < input_channel; j++)
                {
                    for (int p = -center_x; p <= center_x; p++)
                    {
                        if (x - p < 0 || x + p >= input_y || x-p>=input_y || x+p<0)
                        {
                            dx[i][x][y][j] += 0;
                        }
                        else
                        {
                            for (int q = -center_y; q <= center_y; q++)
                            {
                                if (y - q < 0 || y + q >= input_y || y - q >= input_y || y + q < 0)
                                {
                                    dx[i][x][y][j] += 0;
                                }
                                else
                                {
                                    for (int k = 0; k < output_channel; k++)
                                    {
                                        dx[i][x][y][j] += w[ksize_x - 1 - (center_x - p)][ksize_y - 1 - (center_y - q)][j][k] * input[i][x - p][y - q][k];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < batch; i++)
    {
        for (int x = center_x; x < input_x - center_x; x++)
        {
            for (int y = center_y; y < input_y - center_y; y++)
            {
                for (int j = 0; j < input_channel; j++)
                {
                    for (int p = -center_x; p <= center_x; p++)
                    {
                        for (int q = -center_y; q <= center_y; q++)
                        {
                            for (int k = 0; k < output_channel; k++)
                            {
                                dx[i][x][y][j] += w[ksize_x - 1 - (center_x - p)][ksize_y - 1 - (center_y - q)][j][k] * input[i][x - p][y - q][k];
                            }
                        }
                    }
                }
            }
        }
    }
    // 求db
    for (int i = 0; i < output_channel; i++)
    {
        for (int x = 0; x < input_x; x++)
        {
            for (int y = 0; y < input_y; y++)
            {
                for (int z = 0; z < batch; z++)
                {
                    db[i] += input[z][x][y][i];
                }
            }
        }
    }
    // 求dw 先填充再卷积, 与上面前向传播策略一样,不扩充,直接卷积.
    for (int x = 0; x < ksize_x; x++)
    {
        for (int y = 0; y < ksize_y; y++)
        {
            for (int i = 0; i < input_channel; i++)
            {
                for (int j = 0; j < output_channel; j++)
                {
                    for (int m = 0; m < input_x; m++)
                    {
                        int m_x = -center_x + m + x;
                        if (m_x < 0 || m_x >= input_x)
                        {
                            dw[x][y][i][j] += 0;
                        }
                        else
                        {
                            for (int n = 0; n < input_y; n++)
                            {
                                int n_y = -center_y + n + y;
                                if (n_y < 0 || n_y >= input_y)
                                {
                                    dw[x][y][i][j] += 0;
                                }
                                else
                                {
                                    for (int k = 0; k < batch; k++)
                                    {
                                        dw[x][y][i][j] += input[k][m][n][j] * for_input[k][m_x][n_y][i];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void CNN_layer::optim_adam(float lr, float p1 = 0.9, float p2 = 0.99)
{
    t++;
    // get W_S W_b, update w
    float dev_s = 1 - quite_m(p1, t);
    float dev_r = 1 - quite_m(p2, t);
    for (int i = 0; i < ksize_x; i++)
    {
        for (int j = 0; j < ksize_y; j++)
        {
            for (int k = 0; k < input_channel; k++)
            {
                for (int p = 0; p < output_channel; p++)
                {
                    dw[i][j][k][p] /= batch;
                    W_s[i][j][k][p] = p1 * W_s[i][j][k][p] + (1 - p1) * dw[i][j][k][p];
                    W_r[i][j][k][p] = p2 * W_r[i][j][k][p] + (1 - p2) * dw[i][j][k][p] * dw[i][j][k][p];
                    dw[i][j][k][p] = 0;
                    float s_ = W_s[i][j][k][p] / dev_s;
                    float r_ = W_r[i][j][k][p] / dev_r;
                    float u = -lr * s_ / (sqrt(r_) + 1e-8);
                    w[i][j][k][p] += u;
                }
            }
        }
    }
    for (int i = 0; i < output_channel; i++)
    {
        db[i] /= batch;
        B_s[i] = p1 * B_s[i] + (1 - p1) * db[i];
        B_r[i] = p2 * B_r[i] + (1 - p2) * db[i] * db[i];
        db[i] = 0;
        float s_ = B_s[i] / dev_s;
        float r_ = B_r[i] / dev_r;
        float u = -lr * s_ / (sqrt(r_) + 1e-8);
        b[i] += u;
    }
    for (int i = 0; i < batch; i++)
    {
        for (int x = 0; x < input_x; x++)
        {
            for (int y = 0; y < input_y; y++)
            {
                for (int j = 0; j < input_channel; j++)
                {
                    dx[i][x][y][j] = 0;
                }
            }
        }
    }
}


void CNN_layer::optim_SGD(float lr, float p1 = 0.9)
{
    for (int i = 0; i < ksize_x; i++)
    {
        for (int j = 0; j < ksize_y; j++)
        {
            for (int k = 0; k < input_channel; k++)
            {
                for (int p = 0; p < output_channel; p++)
                {
                    dw[i][j][k][p] /= batch;
                    W_s[i][j][k][p] = p1 * W_s[i][j][k][p] - lr * dw[i][j][k][p];
                    dw[i][j][k][p] = 0;
                    w[i][j][k][p] += W_s[i][j][k][p];
                }
            }
        }
    }
    for (int i = 0; i < output_channel; i++)
    {
        db[i] /= batch;
        B_s[i] = p1 * B_s[i] - lr * db[i];
        db[i] = 0;
        b[i] += B_s[i];
    }
    for (int i = 0; i < batch; i++)
    {
        for (int x = 0; x < input_x; x++)
        {
            for (int y = 0; y < input_y; y++)
            {
                for (int j = 0; j < input_channel; j++)
                {
                    dx[i][x][y][j] = 0;
                }
            }
        }
    }
}

void Fully_layer::init(int input_batch, int input_c, int output_c)
{
    t = 0;
    batch = input_batch;
    input_channel = input_c;
    output_channel = output_c;
    out = get_f_mem<float>(batch, output_channel);
    dx = get_f_mem<float>(batch, input_channel);
    w = get_f_mem<float>(input_channel, output_channel);
    init_f_w(w, input_channel, output_channel, MEAN, VARIANCE);
    dw = get_f_mem<float>(input_channel, output_channel);
    w_s = get_f_mem<float>(input_channel, output_channel);
    w_r = get_f_mem<float>(input_channel, output_channel);
    b = new float[output_channel];
    init_b(b, output_channel, MEAN, VARIANCE);
    db = new float[output_channel];
    b_s = new float[output_channel];
    b_r = new float[output_channel];
    for (int i = 0; i < output_channel; i++)
    {
        b_r[i] = db[i] = b_s[i] = 0;
    }
}

void Fully_layer::save_arg(fstream& file)
{
    for(int i=0;i<input_channel; i++)
    {
        for(int j=0; j< output_channel; j++)
        {
            file<<w[i][j]<<" ";
        }
    }
    for(int i=0;i<output_channel; i++)
    {
        file<<b[i]<<" ";
    }
}

void Fully_layer::load_arg(fstream& file)
{
    for(int i=0;i<input_channel; i++)
    {
        for(int j=0; j< output_channel; j++)
        {
            file>>w[i][j];
        }
    }
    for(int i=0;i<output_channel; i++)
    {
        file>>b[i];
    }
}

Fully_layer::~Fully_layer()
{
    delete_f_mem<float>(w, input_channel, output_channel);
    delete_f_mem<float>(dw, input_channel, output_channel);
    delete_f_mem<float>(w_r, input_channel, output_channel);
    delete_f_mem<float>(w_s, input_channel, output_channel);
    delete_f_mem<float>(out, batch, output_channel);
    delete_f_mem<float>(dx, batch, input_channel);
    delete[] b;
    delete[] db;
    delete[] b_s;
    delete[] b_r;
}

void Fully_layer::forward(float **input)
{
    for (int i = 0; i < batch; i++)
    {
        for (int j = 0; j < output_channel; j++)
        {
            out[i][j] = b[j];
            for (int k = 0; k < input_channel; k++)
            {
                out[i][j] += input[i][k] * w[k][j];
            }
        }
    }
}

void Fully_layer::backward(float **input, float **for_input)
{
    // get dx
    for (int i = 0; i < batch; i++)
    {
        for (int j = 0; j < input_channel; j++)
        {
            for (int k = 0; k < output_channel; k++)
            {
                dx[i][j] += input[i][k] * w[j][k];
            }
        }
    }
    // get dw
    for (int i = 0; i < input_channel; i++)
    {
        for (int j = 0; j < output_channel; j++)
        {
            for (int k = 0; k < batch; k++)
            {
                dw[i][j] += for_input[k][i] * input[k][j];
            }
        }
    }
    // get db
    for (int i = 0; i < output_channel; i++)
    {
        for (int k = 0; k < batch; k++)
        {
            db[i] += input[k][i];
        }
    }
}

void Fully_layer::optim_adam(float lr, float p1 = 0.9, float p2 = 0.99)
{
    // update dw_r, dw_s, w;
    t++;
    float dev_s = 1 - quite_m(p1, t);
    float dev_r = 1 - quite_m(p2, t);
    for (int i = 0; i < input_channel; i++)
    {
        for (int j = 0; j < output_channel; j++)
        {
            dw[i][j] /= batch;
            w_s[i][j] = p1 * w_s[i][j] + (1 - p1) * dw[i][j];
            w_r[i][j] = p2 * w_r[i][j] + (1 - p2) * dw[i][j] * dw[i][j];
            dw[i][j] = 0;
            float s_ = w_s[i][j] / dev_s;
            float r_ = w_r[i][j] / dev_r;
            float u = -lr * s_ / (sqrt(r_) + 1e-8);
            w[i][j] += u;
        }
    }
    for (int i = 0; i < output_channel; i++)
    {
        db[i] /= batch;
        b_s[i] = p1 * b_s[i] + (1 - p1) * db[i];
        b_r[i] = p2 * b_r[i] + (1 - p2) * db[i] * db[i];
        db[i] = 0;
        float s_ = b_s[i] / dev_s;
        float r_ = b_r[i] / dev_r;
        float u = -lr * s_ / (sqrt(r_) + 1e-8);
        b[i] += u;
    }
    for (int i = 0; i < batch; i++)
    {
        for (int j = 0; j < input_channel; j++)
        {
            dx[i][j] = 0;
        }
    }
}

void Fully_layer::optim_SGD(float lr, float p1 = 0.9)
{
    // update dw_r, dw_s, w;
    
    for (int i = 0; i < input_channel; i++)
    {
        for (int j = 0; j < output_channel; j++)
        {
            dw[i][j] /= batch;
            w_s[i][j] = p1 * w_s[i][j] - lr * dw[i][j];
            dw[i][j] = 0;
            w[i][j] += w_s[i][j];
        }
    }
    for (int i = 0; i < output_channel; i++)
    {
        db[i] /= batch;
        b_s[i] = p1 * b_s[i] - lr * db[i];
        db[i] = 0;
        b[i] += b_s[i];
    }
    for (int i = 0; i < batch; i++)
    {
        for (int j = 0; j < input_channel; j++)
        {
            dx[i][j] = 0;
        }
    }
}

void ReLU_4D::init(int b, int m, int n, int ch)
{
    batch = b;
    x = m;
    y = n;
    c = ch;
    out = get_mem<float>(batch, x, y, c);
    dx = get_mem<float>(batch, x, y, c);
    d_save = get_mem<float>(batch, x, y, c);
}

ReLU_4D::~ReLU_4D()
{
    delete_mem<float>(out, batch, x, y, c);
    delete_mem<float>(dx, batch, x, y, c);
    delete_mem<float>(d_save, batch, x, y, c);
}

void ReLU_4D::forward(float ****input)
{
    for (int i = 0; i < batch; i++)
    {
        for (int m = 0; m < x; m++)
        {
            for (int n = 0; n < y; n++)
            {
                for (int j = 0; j < c; j++)
                {
                    if (input[i][m][n][j] > 0)
                    {
                        d_save[i][m][n][j] = 1;
                        out[i][m][n][j] = input[i][m][n][j];
                    }
                    else
                    {
                        out[i][m][n][j] = 0;
                        d_save[i][m][n][j] = 0;
                    }
                }
            }
        }
    }
}

void ReLU_4D::backward(float ****input)
{
    for (int i = 0; i < batch; i++)
    {
        for (int m = 0; m < x; m++)
        {
            for (int n = 0; n < y; n++)
            {
                for (int j = 0; j < c; j++)
                {
                    dx[i][m][n][j] = d_save[i][m][n][j] * input[i][m][n][j];
                }
            }
        }
    }
}

void ReLU_2D::init(int b, int ch)
{
    batch = b;
    c = ch;
    out = get_f_mem<float>(batch, c);
    dx = get_f_mem<float>(batch, c);
    d_save = get_f_mem<float>(batch, c);
}

ReLU_2D::~ReLU_2D()
{
    delete_f_mem<float>(out, batch, c);
    delete_f_mem<float>(dx, batch, c);
    delete_f_mem<float>(d_save, batch, c);
}

void ReLU_2D::forward(float **input)
{
    for (int i = 0; i < batch; i++)
    {
        for (int m = 0; m < c; m++)
        {
            if (input[i][m] > 0)
            {
                d_save[i][m] = 1;
                out[i][m] = input[i][m];
            }
            else
            {
                out[i][m] = 0;
                d_save[i][m] = 0;
            }
        }
    }
}

void ReLU_2D::backward(float **input)
{
    for (int i = 0; i < batch; i++)
    {
        for (int m = 0; m < c; m++)
        {
            dx[i][m] = d_save[i][m] * input[i][m];
        }
    }
}

void MaxPooling::init(int b, int input_m, int input_n, int c, int ksize_m, int ksize_n, int strides_m = -1, int strides_n = -1)
{
    batch = b;
    input_x = input_m;
    input_y = input_n;
    intput_channel = c;
    ksize_x = ksize_m;
    ksize_y = ksize_n;
    if (strides_m == -1)
    {
        strides_x = ksize_m;
        strides_y = ksize_n;
        output_x = int(input_x / ksize_x);
        output_y = int(input_y / ksize_y);
    }
    else
    {
        strides_x = strides_m;
        strides_y = strides_n;
        output_x = int(1 + (input_x - ksize_x) / strides_x);
        output_y = int(1 + (input_y - ksize_y) / strides_y);
    }
    local_x = new int[output_x * output_y * intput_channel*batch];
    local_y = new int[output_x * output_y * intput_channel*batch];
    out = get_mem<float>(batch, output_x, output_y, intput_channel);
    dx = get_mem<float>(batch, input_x, input_y, intput_channel);
}

MaxPooling::~MaxPooling()
{
    delete_mem<float>(out, batch, output_x, output_y, intput_channel);
    delete_mem<float>(dx, batch, input_x, input_y, intput_channel);
    delete[] local_x;
    delete[] local_y;
}

void MaxPooling::forward(float ****input)
{
    int count = 0;
    for (int i = 0; i < batch; i++)
    {
        for (int x = 0; x < output_x; x++)
        {
            for (int y = 0; y < output_y; y++)
            {
                for (int j = 0; j < intput_channel; j++)
                {
                    out[i][x][y][j] = input[i][strides_x * x][strides_y * y][j];
                    local_x[count] = strides_x * x;
                    local_y[count] = strides_y * y;
                    for (int p = 0; p < ksize_x; p++)
                    {
                        for (int q = 0; q < ksize_y; q++)
                        {
                            if (out[i][x][y][j] < input[i][strides_x * x + p][strides_y * y + q][j])
                            {
                                out[i][x][y][j] = input[i][strides_x * x + p][strides_y * y + q][j];
                                local_x[count] = strides_x * x + p;
                                local_y[count] = strides_y * y + q;
                            }
                        }
                    }
                    count++;
                }
            }
        }
    }
}

void MaxPooling::backward(float ****input)
{
    for(int i=0;i<batch;i++)
    {
        for(int x=0; x<input_x; x++)
        {
            for(int y=0; y<input_y;y++)
            {
                for(int j=0;j<intput_channel;j++)
                {
                    dx[i][x][y][j] = 0;
                }
            }
        }
    }
    int count = 0;
    for (int i = 0; i < batch; i++)
    {
        for (int x = 0; x < output_x; x++)
        {
            for (int y = 0; y < output_y; y++)
            {
                for (int j = 0; j < intput_channel; j++)
                {
                    dx[i][local_x[count]][local_y[count]][j] = input[i][x][y][j];
                    count++;
                }
            }
        }
    }
}

void DropOut::init(int b, int c, float pre)
{
    batch = b;
    channel = c;
    k_pre = pre;
    dx = get_f_mem<float>(batch, channel);
    out = get_f_mem<float>(batch, channel);
    d_save = get_f_mem<float>(batch, channel);
}

DropOut::~DropOut()
{
    delete_f_mem<float>(dx, batch, channel);
    delete_f_mem<float>(out, batch, channel);
    delete_f_mem<float>(d_save, batch, channel);
}

void DropOut::forward(float **input)
{
    srand(time(0));
    for (int i = 0; i < batch; i++)
    {
        for (int j = 0; j < channel; j++)
        {
            float w = rand() % 100 / float(99);
            if (w < k_pre)
            {
                out[i][j] = input[i][j] / k_pre;
                d_save[i][j] = 1.0 / k_pre;
            }
            else
            {
                out[i][j] = 0;
                d_save[i][j] = 0;
            }
        }
    }
}

void DropOut::backward(float **input)
{
    for (int i = 0; i < batch; i++)
    {
        for (int j = 0; j < channel; j++)
        {
            dx[i][j] =d_save[i][j] * input[i][j];
        }
    }
}

void Reshape::init(int b, int input_m, int input_n, int input_ch)
{
    batch = b;
    input_x = input_m;
    input_y = input_n;
    input_c = input_ch;
    output_c = input_x * input_y * input_c;
    dx = get_mem<float>(batch, input_x, input_y, input_c);
    out = get_f_mem<float>(batch, output_c);
}

Reshape::~Reshape()
{
    delete_f_mem<float>(out, batch, output_c);
    delete_mem<float>(dx, batch, input_x, input_y, input_c);
}

void Reshape::forward(float ****input)
{
    for (int i = 0; i < batch; i++)
    {
        for (int x = 0; x < input_x; x++)
        {
            for (int y = 0; y < input_y; y++)
            {
                for (int j = 0; j < input_c; j++)
                {
                    out[i][x * input_y * input_c + y * input_c + j] = input[i][x][y][j];
                }
            }
        }
    }
}

void Reshape::backward(float **input)
{
    for (int i = 0; i < batch; i++)
    {
        for (int x = 0; x < input_x; x++)
        {
            for (int y = 0; y < input_y; y++)
            {
                for (int j = 0; j < input_c; j++)
                {
                    dx[i][x][y][j] = input[i][x * input_y * input_c + y * input_c + j];
                }
            }
        }
    }
}

void cat::init(int b, int input1_ch, int input2_ch)
{
    batch = b;
    input1_c = input1_ch;
    input2_c = input2_ch;
    output_c = input1_c+input2_c;
    out = get_f_mem<float>(batch, input1_c + input2_c);
    dx1 = get_f_mem<float>(batch, input1_c);
    dx2 = get_f_mem<float>(batch, input2_c);
}

cat::~cat()
{
    delete_f_mem<float>(out, batch, input1_c + input2_c);
    delete_f_mem<float>(dx1, batch, input1_c);
    delete_f_mem<float>(dx2, batch, input2_c);
}

void cat::forward(float **input1, float **input2)
{
    for (int i = 0; i < batch; i++)
    {
        for (int j = 0; j < input1_c; j++)
        {
            out[i][j] = input1[i][j];
        }
        for (int j = 0; j < input2_c; j++)
        {
            out[i][j + input1_c] = input2[i][j];
        }
    }
}

void cat::backward(float **input)
{
    for (int i = 0; i < batch; i++)
    {
        for (int j = 0; j < input1_c; j++)
        {
            dx1[i][j] = input[i][j];
        }
        for (int j = 0; j < input2_c; j++)
        {
            dx2[i][j] = input[i][input1_c + j];
        }
    }
}

void softmax::init(int b, int inpuch_c)
{
    batch = b;
    input_channel = inpuch_c;
    dx = get_f_mem<float>(batch, input_channel);
    out = get_f_mem<float>(batch, input_channel);
}

softmax::~softmax()
{
    delete_f_mem<float>(dx, batch, input_channel);
    delete_f_mem<float>(out, batch, input_channel);
}

void softmax::forward(float **input)
{
    for(int i=0;i<batch;i++)
    {
        float sum = 0;
        for(int j=0;j<input_channel;j++)
        {
            sum += exp(input[i][j]+0.5);
        }
        for(int j=0;j<input_channel;j++)
        {
            out[i][j] = exp(input[i][j]+0.5) / sum;
        }
    }
}

void softmax::backward(float **input)
{
    for(int i=0;i<batch;i++)
    {
        for(int j=0;j<input_channel;j++)
        {
            dx[i][j] = 0;
            for(int k=0;k<input_channel;k++)
            {
                if(j==k)
                {
                    dx[i][j] += input[i][k]*(out[i][j]-out[i][j]*out[i][j]);
                }
                else
                {
                    dx[i][j] -= input[i][k]*out[i][j]*out[i][k];
                }
            }
        }
    }
}

void loss::init(int b, int input_c)
{
    batch = b;
    input_channel = input_c;
    dx = get_f_mem<float>(batch, input_channel);
}

loss::~loss()
{
    delete_f_mem<float>(dx, batch, input_channel);
}

void loss::forward(float **y_pre, int *y_true)
{
    Loss = 0;
    for(int i=0;i<batch;i++)
    {
        // avoid y_pre[i][y_true[i]]=0 , add 1e-8
        Loss -= log(y_pre[i][y_true[i]] + 1e-8);
    }
    Loss /= batch;
}

void loss::backward(float **y_pre, int *y_true)
{
    for(int i=0;i<batch;i++)
    {
        for(int j=0;j<input_channel;j++)
        {
            dx[i][j] = 0;
        }
    }
    for(int i=0;i<batch;i++)
    {
        dx[i][y_true[i]] = -1.0/(y_pre[i][y_true[i]]+1e-8)/batch;
    }
}

float accurate(float **y_pre, int *y_true, int batch, int classes)
{
    int num=0;
    for(int i=0;i<batch;i++)
    {
        int max = 0;
        for(int j=1;j<classes;j++)
        {
            if(y_pre[i][j]>y_pre[i][max])
            {
                max = j;
            }
        }
        if(max == y_true[i])
        {
            num += 1;
        }
    }
    return num*1.0/batch;
}