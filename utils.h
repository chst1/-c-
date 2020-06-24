#include<random>
#include<string>
extern float MEAN;
extern float VARIANCE;


float quite_m(float a, int n);

void init_w(float ****w, int batch, int m, int n, int c,float mean,float variance);

void init_f_w(float **w, int batch, int channel, int mean, int variance);

void init_b(float *b, int c, float mean, float variance);
