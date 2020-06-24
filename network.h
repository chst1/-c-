#include"utils.h"
#include"layer.h"
class network
{
   // layer1
   CNN_layer conv2d1;
   ReLU_4D relu4d1;
   MaxPooling maxpooling1;
   // layer2
   CNN_layer conv2d2;
   ReLU_4D relu4d2;
   // layer3
   CNN_layer conv2d3;
   ReLU_4D relu4d3;
   // layer4
   CNN_layer conv2d4;
   ReLU_4D relu4d4;
   MaxPooling maxpooling4;
   // layer5
   CNN_layer conv2d5;
   ReLU_4D relu4d5;
   // layer6
   CNN_layer conv2d6;
   ReLU_4D relu4d6;
   // layer7
   CNN_layer conv2d7;
   ReLU_4D relu4d7;
   MaxPooling maxpooling7;
   // layer8
   CNN_layer conv2d8;
   ReLU_4D relu4d8;
   // layer9
   CNN_layer conv2d9;
   ReLU_4D relu4d9;
   // lay10
   CNN_layer conv2d10;
   ReLU_4D relu4d10;
   MaxPooling maxpooling10;
   // reshape
   Reshape reshape1;
   Reshape reshape2;
   // cat
   cat concat;
   // fully_layer
   Fully_layer fully1;
   ReLU_2D relu2d;
   // dropout
   DropOut dropout;
   // SVM
   Fully_layer svm;
   // softmax
   softmax softmax1;
   // loss
   loss Loss;
   float acc;
public:
   network(int batch, int image_x, int image_y, int image_c);
   // ~network();
   void forward(float ****image, int *y_true);
   void backward(float ****image, int *y_true);
   void optim(float lr, float p1, float p2);
   float get_loss();
   float get_accurate();
};