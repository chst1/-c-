# 各个文件作用

getdata.py通过视频文件获取数据

layer.cpp/layer.cpp网络各个层具体实现

load_data.h/load_data.cpp装载数据

utils.cpp/utils.h网络参数初始化

utils_other.cpp申请存储和释放存储

train.cpp训练

test.cpp测试

test_camrea.cpp使用摄像头测试

# 执行方式

首先将视频数据和文件进行更改名字，全部文件夹变成１，２，３，４，５，６。视频变成1.mp4，２.mp4，...

更改getdata.py中路径名和存储图片的路径名和两个txt路径名，运行getdata.py获取数据集。

更改train.cpp中路径名，运行，获取model。

更改test.cpp中路径名，运行，测试效果。

更改test_camrea.cpp路径名，运行，测试效果。

各个文件获取可执行文件方式在最后一行的注释中。
=======
　　由于直接在github上写公式会无法正常显示, 因此我将.md文件转换为pdf文件, 详情请阅览pdf文件, 带来的不便请见谅. 文章代码均为原创, 转载请注明出处. 谢谢!