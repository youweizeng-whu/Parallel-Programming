#include <cuda_runtime.h>

// 读取
uchar4 *read_png(const char* filename, int *width, int *height);
// 将图像数据写入 PNG 文件的函数声明
void write_png(const uchar4 *image, int width, int height, const char* filename);
