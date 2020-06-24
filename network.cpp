#include"network.h"
network::network(int batch, int image_x, int image_y, int image_c)
{
    conv2d1.init(batch, image_x, image_y, image_c, 6, 3, 3, 1, 1);
    relu4d1.init(batch, image_x, image_y, image_c);
    maxpooling1.init(batch, image_x, image_y, image_c, 2, 2, 2, 2);
    conv2d2.init(batch, maxpooling1.output_x, maxpooling1.output_y, 6, 6, 3, 3, 1, 1);
    relu4d2.init(batch, maxpooling1.output_x, maxpooling1.output_y, 6);
    conv2d3.init(batch, conv2d2.input_x, conv2d2.input_y, 6, 6, 3, 3, 1, 1);
    relu4d3.init(batch, conv2d3.input_x, conv2d3.input_y, conv2d3.output_channel);
    conv2d4.init(batch, relu4d3.x, relu4d3.y, relu4d3.c, 12, 3, 3, 1, 1);
    relu4d4.init(batch, conv2d4.input_x, conv2d4.input_y, conv2d4.output_channel);
    maxpooling4.init(batch, relu4d4.x, relu4d4.y, relu4d4.c, 2, 2, 2, 2);
    conv2d5.init(batch, maxpooling4.output_x, maxpooling4.output_y, maxpooling4.intput_channel, 12, 3, 3, 1, 1);
    relu4d5.init(batch, conv2d5.input_x, conv2d5.input_y, conv2d5.output_channel);
    conv2d6.init(batch, relu4d5.x, relu4d5.y, relu4d5.c, 12, 3, 3, 1, 1);
    relu4d6.init(batch, conv2d5.input_x, conv2d5.input_y, conv2d5.output_channel);
    conv2d7.init(batch, relu4d6.x, relu4d6.y, relu4d6.c, 24, 3, 3, 1, 1);
    relu4d7.init(batch, conv2d7.input_x, conv2d7.input_y, conv2d7.output_channel);
    maxpooling7.init(batch, relu4d7.x, relu4d7.y, relu4d7.c, 2, 2, 2, 2);
    conv2d8.init(batch, maxpooling7.output_x, maxpooling7.output_y, maxpooling7.intput_channel, 24, 3, 3, 1, 1);
    relu4d8.init(batch, conv2d8.input_x, conv2d8.input_y, conv2d8.output_channel);
    conv2d9.init(batch, relu4d8.x, relu4d8.y, relu4d8.c, 24, 3, 3, 1, 1);
    relu4d9.init(batch, conv2d9.input_x, conv2d9.input_y, conv2d9.output_channel);
    conv2d10.init(batch, relu4d9.x, relu4d9.y, relu4d9.c, 48, 3, 3, 1, 1);
    relu4d10.init(batch, conv2d10.input_x, conv2d10.input_y, conv2d10.output_channel);
    maxpooling10.init(batch, relu4d10.x, relu4d10.y, relu4d10.c, 2, 2, 2, 2);
    reshape1.init(batch, maxpooling7.output_x, maxpooling7.output_y, maxpooling7.intput_channel);
    reshape2.init(batch, maxpooling10.output_x, maxpooling10.output_y, maxpooling10.intput_channel);
    concat.init(batch, reshape1.output_c, reshape2.output_c);
    fully1.init(batch, concat.output_c, 3072);
    relu2d.init(batch, fully1.output_channel);
    dropout.init(batch, relu2d.c, 0.5);
    svm.init(batch, dropout.channel, 69);
    softmax1.init(batch, 69);
    Loss.init(batch, 69);
}

void network::forward(float ****image, int *y_true)
{
   conv2d1.forward(image);
   relu4d1.forward(conv2d1.out);
   maxpooling1.forward(relu4d1.out);
   conv2d2.forward(maxpooling1.out);
   relu4d2.forward(conv2d2.out);
   conv2d3.forward(relu4d2.out);
   relu4d3.forward(conv2d3.out);
   conv2d4.forward(relu4d3.out);
   relu4d4.forward(conv2d4.out);
   maxpooling4.forward(relu4d4.out);
   conv2d5.forward(maxpooling4.out);
   relu4d5.forward(conv2d5.out);
   conv2d6.forward(relu4d5.out);
   relu4d6.forward(conv2d6.out);
   conv2d7.forward(relu4d6.out);
   relu4d7.forward(conv2d7.out);
   maxpooling7.forward(relu4d7.out);
   conv2d8.forward(maxpooling7.out);
   relu4d8.forward(conv2d8.out);
   conv2d9.forward(relu4d8.out);
   relu4d9.forward(conv2d9.out);
   conv2d10.forward(relu4d9.out);
   relu4d10.forward(conv2d10.out);
   maxpooling10.forward(relu4d10.out);
   reshape1.forward(maxpooling7.out);
   reshape2.forward(maxpooling10.out);
   concat.forward(reshape1.out, reshape2.out);
   fully1.forward(concat.out);
   relu2d.forward(fully1.out);
   dropout.forward(relu2d.out);
   svm.forward(dropout.out);
   softmax1.forward(svm.out);
   Loss.forward(softmax1.out, y_true);
   acc = accurate(softmax1.out, y_true, softmax1.batch, softmax1.input_channel);
}

void network::backward(float ****image, int *y_true)
{
    Loss.backward(softmax1.out, y_true);
    softmax1.backward(Loss.dx);
    svm.backward(softmax1.dx, dropout.out);
    dropout.backward(svm.dx);
    relu2d.backward(dropout.dx);
    fully1.backward(relu2d.dx, concat.out);
    concat.backward(fully1.dx);
    reshape1.backward(concat.dx1);
    maxpooling7.backward(reshape1.dx);
    relu4d7.backward(maxpooling7.dx);
    conv2d7.backward(relu4d7.dx, relu4d6.out);
    reshape2.backward(concat.dx2);
    maxpooling10.backward(reshape2.dx);
    relu4d10.backward(maxpooling10.dx);
    conv2d10.backward(relu4d10.dx, relu4d9.out);
    relu4d9.backward(conv2d10.dx);
    conv2d9.backward(relu4d9.dx, conv2d8.out);
    relu4d8.backward(conv2d10.dx);
    conv2d8.backward(relu4d8.dx, maxpooling7.out);
    maxpooling7.backward(conv2d8.dx);
    relu4d7.backward(maxpooling7.dx);
    conv2d7.backward(relu4d7.dx, relu4d6.out);
    relu4d6.backward(conv2d7.dx);
    conv2d6.backward(relu4d6.dx, relu4d5.out);
    relu4d5.backward(conv2d6.dx);
    conv2d5.backward(relu4d5.dx, maxpooling4.out);
    maxpooling4.backward(conv2d5.dx);
    relu4d4.backward(maxpooling4.dx);
    conv2d4.backward(relu4d4.dx, relu4d3.out);
    relu4d3.backward(conv2d4.dx);
    conv2d3.backward(relu4d3.dx, relu4d2.out);
    relu4d2.backward(conv2d3.dx);
    conv2d2.backward(relu4d2.dx, maxpooling1.out);
    maxpooling1.backward(conv2d2.dx);
    relu4d1.backward(maxpooling1.dx);
    conv2d1.backward(relu4d1.dx, image);
}

void network::optim(float lr, float p1, float p2)
{
    conv2d1.optim(lr, p1, p2);
    conv2d2.optim(lr, p1, p2);
    conv2d3.optim(lr, p1, p2);
    conv2d4.optim(lr, p1, p2);
    conv2d5.optim(lr, p1, p2);
    conv2d6.optim(lr, p1, p2);
    conv2d7.optim(lr, p1, p2);
    conv2d8.optim(lr, p1, p2);
    conv2d9.optim(lr, p1, p2);
    conv2d10.optim(lr, p1, p2);
    fully1.optim(lr, p1, p2);
    svm.optim(lr, p1, p2);
}

float network::get_accurate()
{
    return acc;
}

float network::get_loss()
{
    return Loss.Loss;
}