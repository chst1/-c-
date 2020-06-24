#include"utils.h"
#include<random>
float MEAN = 0.0;
float VARIANCE = 0.1;

float quite_m(float a, int n)
{
    float count = 1.0, m = a;
    while(n)
    {
        if(n%2)
        {
            count *= m;
            m = m*m;
            n = int(n/2);
        }
        else
        {
            m = m*m;
            n = int(n/2);
        }
    }
    return count;
}

void init_w(float ****w, int batch, int m, int n, int c,float mean,float variance)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(mean, variance);
    for(int i=0;i<batch;i++)
    {
        for(int j=0;j<m;j++)
        {
            for(int k=0;k<n;k++)
            {
                for(int p=0;p<c;p++)
                {
                    w[i][j][k][p] = d(gen);
                }
            }
        }
    }
}

void init_f_w(float **w, int batch, int channel, int mean, int variance)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(mean, variance);
    for(int i=0;i<batch; i++)
    {
        for(int j=0;j<channel;j++)
        {
            w[i][j] = d(gen);
        }
    }
}

void init_b(float *b, int c, float mean, float variance)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(mean, variance);
    for(int i=0;i<c;i++)
    {
        b[i] = d(gen);
    }
}