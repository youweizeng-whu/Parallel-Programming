#include "png_wr.h"
#include <iostream>

__global__ void processImageKernel(uchar4 *d_imageData, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int rTotal = 0;
    int gTotal = 0;
    int bTotal = 0;
    int count = 0;

    // 遍历 5x5 的格子
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int newX = x + dx;
            int newY = y + dy;

            // 检查边界
            if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                uchar4 pixel = d_imageData[newY * width + newX];
                rTotal += pixel.x;
                gTotal += pixel.y;
                bTotal += pixel.z;
                count++;
            }
        }
    }
    __syncthreads();

    // 计算平均值
    uchar4 outputPixel;
    outputPixel.x = rTotal / count;
    outputPixel.y = gTotal / count;
    outputPixel.z = bTotal / count;
    outputPixel.w = 255; // Alpha 值保持不变

    // 将计算后的像素写回图像
    d_imageData[y * width + x] = outputPixel;
}

__global__ void processImageKernel2(uchar4 *d_imageData, uchar4 *d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int rTotal = 0;
    int gTotal = 0;
    int bTotal = 0;
    int count = 0;

    // 遍历 5x5 的格子
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            int newX = x + dx;
            int newY = y + dy;

            // 检查边界
            if (newX >= 0 && newX < width && newY >= 0 && newY < height) {
                uchar4 pixel = d_imageData[newY * width + newX];
                rTotal += pixel.x;
                gTotal += pixel.y;
                bTotal += pixel.z;
                count++;
            }
        }
    }

    // 计算平均值
    uchar4 outputPixel;
    outputPixel.x = rTotal / count;
    outputPixel.y = gTotal / count;
    outputPixel.z = bTotal / count;
    outputPixel.w = 255; // Alpha 值保持不变

    // 将计算后的像素写入 d_output
    d_output[y * width + x] = outputPixel;
}

__global__ void processImageKernel3(uchar4 *d_imageData, uchar4 *d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int localX = threadIdx.x + 1;
    int localY = threadIdx.y + 1;
    // 定义共享内存
    __shared__ uchar4 sharedPixels[18][18];

    // 检查全局边界
    if (x < width && y < height) {
        // 将像素值加载到共享内存
        sharedPixels[localY][localX] = d_imageData[y * width + x];
    }

    // 处理边界像素
    // 左边界
    if (threadIdx.x < 1 && x > 0) {
        sharedPixels[localY][0] = d_imageData[y * width + (x - 1)];
        // 左上角
        if (threadIdx.y < 1 && y > 0)
            sharedPixels[0][0] = d_imageData[(y - 1) * width + (x - 1)];
        // 左下角
        if (threadIdx.y >= blockDim.y - 1 && y < height - 1)
            sharedPixels[18 - 1][0] = d_imageData[(y + 1) * width + (x - 1)];
    }
    // 右边界
    if (threadIdx.x >= blockDim.x - 1 && x < width - 1) {
        sharedPixels[localY][18 - 1] = d_imageData[y * width + (x + 1)];
        // 右上角
        if (threadIdx.y < 1 && y > 0)
            sharedPixels[0][18 - 1] = d_imageData[(y - 1) * width + (x + 1)];
        // 右下角
        if (threadIdx.y >= blockDim.y - 1 && y < height - 1)
            sharedPixels[18 - 1][18 - 1] = d_imageData[(y + 1) * width + (x + 1)];
    }
    // 上边界
    if (threadIdx.y < 1 && y > 0) {
        sharedPixels[0][localX] = d_imageData[(y - 1) * width + x];
    }
    // 下边界
    if (threadIdx.y >= blockDim.y - 1 && y < height - 1) {
        sharedPixels[18 - 1][localX] = d_imageData[(y + 1) * width + x];
    }

    // 等待所有线程完成数据加载
    __syncthreads();

    // 计算平均值
    int rTotal = 0;
    int gTotal = 0;
    int bTotal = 0;
    int count = 0;
    if (x < width && y < height) {

        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                if (localY + dy < 0 || localY + dy > 17 || localX + dx < 0 || localX + dx > 17) {
                    continue;
                }
                uchar4 pixel = sharedPixels[localY + dy][localX + dx];
                rTotal += pixel.x;
                gTotal += pixel.y;
                bTotal += pixel.z;
                count++;
            }
        }

        uchar4 outputPixel;
        outputPixel.x = rTotal / count;
        outputPixel.y = gTotal / count;
        outputPixel.z = bTotal / count;
        outputPixel.w = 255; 

        // 将计算后的像素写入 d_output
        d_output[y * width + x] = outputPixel;
    }

}


// processImageKernel2:Total time for 1000 executions: 105.04 ms
// processImageKernel3:Total time for 1000 executions: 92.7334 ms
void time_test1(uchar4 *d_imageData, int width, int height) {
    // 为 d_output 申请内存
    uchar4 *d_output;
    size_t numBytes = width * height * sizeof(uchar4);
    cudaError_t cudaStatus = cudaMalloc((void**)&d_output, numBytes);

    if (cudaStatus != cudaSuccess) {
        // cudaMalloc 失败的处理
        fprintf(stderr, "cudaMalloc failed for d_output!");
        // 进一步的错误处理代码...
    }
    // 假设的 kernel 函数调用
    dim3 dimBlock(16, 16); // 可以根据需求调整
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    // ... 使用 d_output ...
    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start);

    // 执行 kernel 1000 次
    for (int i = 0; i < 1000; ++i) {
        // processImageKernel2<<<dimGrid, dimBlock>>>(d_imageData, d_output, width, height);
        processImageKernel3<<<dimGrid, dimBlock>>>(d_imageData, d_output, width, height);
    }


    // 记录结束时间
    cudaEventRecord(stop);

    // 同步事件
    cudaEventSynchronize(stop);

    // 计算经过时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Total time for 1000 executions: " << milliseconds << " ms" << std::endl;
    // 为 h_imageData 分配内存
    uchar4 *h_imageData = new uchar4[width * height];

    cudaMemcpy(h_imageData, d_output, numBytes, cudaMemcpyDeviceToHost);

    // 写入图像
    write_png(h_imageData, width, height, "1000kernel_test.png");
    // 销毁 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);    
    
    // 最后，不要忘记释放内存
    cudaFree(d_output);   
    delete[] h_imageData; 
}

int main() {
    const char* inputFilename = "input.png";
    const char* outputFilename = "output.png";

    int width, height;
    uchar4 *h_imageData;
    uchar4 *d_imageData;

    // 读取图像
    h_imageData = read_png(inputFilename, &width, &height);
    if (h_imageData == nullptr) {
        std::cerr << "读取图像失败: " << inputFilename << std::endl;
        return 1;
    }

    std::cout << "图像读取成功: " << inputFilename << std::endl;
    std::cout << "图像尺寸: " << width << " x " << height << std::endl;

    size_t numBytes = width * height * sizeof(uchar4);
    cudaMalloc(&d_imageData, numBytes);
    cudaMemcpy(d_imageData, h_imageData, numBytes, cudaMemcpyHostToDevice);

    // 假设的 kernel 函数调用
    dim3 dimBlock(16, 16); // 可以根据需求调整
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

    processImageKernel<<<dimGrid, dimBlock>>>(d_imageData, width, height);

    time_test1(d_imageData, width, height);

    cudaMemcpy(h_imageData, d_imageData, numBytes, cudaMemcpyDeviceToHost);

    // 写入图像
    write_png(h_imageData, width, height, outputFilename);
    std::cout << "图像写入成功: " << outputFilename << std::endl;

    // 释放图像数据内存（假设read_png分配了内存）
    free(h_imageData);
    cudaFree(d_imageData);

    return 0;
}
