#include<random>
template<class T>
T ****get_mem(int batch, int m, int n, int c)
{
    T ****a = new T ***[batch];
    for(int i=0;i<batch;i++)
    {
        a[i] = new T **[m];
        for(int j=0;j<m;j++)
        {
            a[i][j] = new T *[n];
            for(int k=0;k<n;k++)
            {
                a[i][j][k] = new T [c];
                for(int p=0;p<c;p++)
                {
                    a[i][j][k][p] = 0;
                }
            }
        }
    }
    return a;
}

template<class T> 
T **get_f_mem(int batch, int c)
{
    T **a = new T *[batch];
    for(int i=0;i<batch;i++)
    {
        a[i] = new T [c];
        for(int j=0;j<c;j++)
        {
            a[i][j] = 0;
        }
    }
    return a;
}

template<class T>
void delete_mem(T ****a, int batch, int m, int n, int c)
{
    for(int i=0;i<batch;i++)
    {
        for(int j=0;j<m;j++)
        {
            for(int k=0;k<n;k++)
            {
                delete [] a[i][j][k];
            }
            delete a[i][j];
        }
        delete a[i];
    }
    delete [] a;
}

template<class T>
void delete_f_mem(T **a, int batch, int c)
{
    for(int i=0;i<batch;i++)
    {
        delete [] a[i];
    }
    delete [] a;
}