#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <cuda_runtime.h>  // 或者 #include <vector_types.h>


uchar4 *read_png(const char* filename, int *width, int *height) {
    FILE *fp = fopen(filename, "rb");
    if(!fp) {
        perror("File opening failed");
        return NULL;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if(!png) abort();

    png_infop info = png_create_info_struct(png);
    if(!info) abort();

    if(setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);

    png_read_info(png, info);

    *width      = png_get_image_width(png, info);
    *height     = png_get_image_height(png, info);
    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth  = png_get_bit_depth(png, info);

    // Read any color_type into 8bit depth, RGBA format.
    // See http://www.libpng.org/pub/png/libpng-manual.txt

    if(bit_depth == 16)
        png_set_strip_16(png);

    if(color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
    if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if(png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    // These color_type don't have an alpha channel then fill it with 0xff.
    if(color_type == PNG_COLOR_TYPE_RGB ||
       color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if(color_type == PNG_COLOR_TYPE_GRAY ||
       color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);

    png_read_update_info(png, info);

    png_bytep *row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * *height);
    for(int y = 0; y < *height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png,info));
    }

    png_read_image(png, row_pointers);

    uchar4 *image = (uchar4 *)malloc(*width * *height * sizeof(uchar4));

    for(int y = 0; y < *height; y++) {
        png_bytep row = row_pointers[y];
        for(int x = 0; x < *width; x++) {
            png_bytep px = &(row[x * 4]);
            image[y * *width + x].x = px[0];
            image[y * *width + x].y = px[1];
            image[y * *width + x].z = px[2];
            image[y * *width + x].w = px[3];
        }
        free(row_pointers[y]);
    }
    free(row_pointers);

    png_destroy_read_struct(&png, &info, NULL);
    fclose(fp);

    return image;
}



void saveGreyImageToPNG(unsigned char* h_greyImage, const char* filename, size_t width, size_t height) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Could not open file %s for writing\n", filename);
        return;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "Could not allocate write struct\n");
        fclose(fp);
        return;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "Could not allocate info struct\n");
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        fclose(fp);
        return;
    }

    // Set up error handling.
    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error during png creation\n");
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return;
    }

    png_init_io(png_ptr, fp);

    // Write header (8 bit colour depth)
    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    // Write image data
    for (size_t y = 0; y < height; y++) {
        png_write_row(png_ptr, h_greyImage + y * width);
    }

    // End write
    png_write_end(png_ptr, NULL);

    // Cleanup
    if (fp != NULL) fclose(fp);
    if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
}

// 定义一个 CUDA 核函数
__global__ void rgba_to_greyscale(const uchar4 * const rgbaImage,
    unsigned char* const greyImage,int numRows, int numCols)
{
    // 计算每个线程应该处理的像素位置
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 确保线程不会处理图像边界之外的像素
    if (x < numCols && y < numRows) {
        // 计算一维索引
        int index = y * numCols + x;

        // 使用 index 来访问和处理 rgbaImage 中的像素
        uchar4 rgba = rgbaImage[index];
        // ... 对 pixel 进行处理 ...
        unsigned char grey = static_cast<unsigned char>(0.299f * rgba.x + 0.587f * rgba.y + 0.114f * rgba.z);
        // 将计算得到的灰度值存储到输出图像
        greyImage[index] = grey;
    }
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
    unsigned char* const d_greyImage, size_t numRows, size_t numCols){
        // 将数据从主机内存复制到设备内存
        size_t numBytes = numRows * numCols * sizeof(uchar4);
        cudaMemcpy(d_rgbaImage, h_rgbaImage, numBytes, cudaMemcpyHostToDevice);
        const dim3 blockSize(16, 16);
        // 确保整个 grid 能够覆盖整个图像, 但是要注意越界
        const dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x,
         (numRows + blockSize.y - 1) / blockSize.y);
         // (87, 55)
        rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
        cudaDeviceSynchronize();
}

// sh run.sh 执行指令
int main() {
    int width, height;
    const char* filename = "WX20240120-133529@2x.png";
    uchar4 *h_rgbaImage = read_png(filename, &width, &height);

    if (h_rgbaImage == NULL) {
        printf("Error reading PNG file!\n");
        return EXIT_FAILURE;
    }

    uchar4 *d_rgbaImage;

    // 计算需要分配的总字节数
    size_t numBytes = width * height * sizeof(uchar4);

    // 使用 cudaMalloc 为图像分配内存
    cudaError_t err = cudaMalloc((void **)&d_rgbaImage, numBytes);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    unsigned char* d_greyImage; // 注意：没有 const，因为我们需要修改这个指针

    // 计算需要分配的总字节数
    numBytes = width * height * sizeof(unsigned char);

    // 使用 cudaMalloc 为图像分配内存
    err = cudaMalloc((void **)&d_greyImage, numBytes);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, height, width);

    // 分配主机内存
    unsigned char* h_greyImage = (unsigned char*)malloc(numBytes);
    // 从 GPU 内存复制到主机内存
    cudaMemcpy(h_greyImage, d_greyImage, numBytes, cudaMemcpyDeviceToHost);

    // 保存图像为 PNG
    saveGreyImageToPNG(h_greyImage, "output.png", width, height);

    // 完成后，释放内存
    free(h_greyImage);
    cudaFree(d_greyImage);
    cudaFree(d_rgbaImage);
    return 0;
}
