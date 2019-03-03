#include <iostream>
#include <random>
#include <string>
#include "utils.h"
#include "network.h"
#include "utils_other.h"
#include "load_data.h"
using namespace std;
int main()
{
    network net(10, 96, 96, 3);
    int *y_true = new int[10];
    float ****image = get_mem<float>(10, 96, 96, 3);
    load_data train_file("train.txt", 10, 96, 96, 3);
    for (int i = 0; i < 100; i++)
    {
        train_file.next_batch(image, y_true);
        net.forward(image, y_true);
        cout<<i<<"th time is training"<<endl<<net.get_loss()<<endl<<net.get_accurate()<<endl;
        cout<<endl;
        net.backward(image, y_true);
        net.optim(0.001, 0.9, 0.99);
    }
    return 0;
}