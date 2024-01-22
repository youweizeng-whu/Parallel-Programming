#!/bin/bash

# 编译 blurr.cu
nvcc -c blurr.cu -o blurr.o

# 编译 png_wr.cu
nvcc -c png_wr.cu -o png_wr.o

# 链接对象文件并生成最终的可执行文件
nvcc blurr.o png_wr.o -lpng -o my_program

# 删除所有的.o文件
rm *.o

# 运行程序
./my_program
