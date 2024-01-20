#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <cuda_runtime.h>  // 或者 #include <vector_types.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

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

// 灰度转换函数
void rgba_to_greyscale_parallel(const uchar4 *h_rgbaImage, unsigned char *h_greyImage, int width, int height, int startY, int endY) {
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            uchar4 rgba = h_rgbaImage[index];
            unsigned char grey = static_cast<unsigned char>(0.299f * rgba.x + 0.587f * rgba.y + 0.114f * rgba.z);
            h_greyImage[index] = grey;
        }
    }
}

void rgba_to_greyscale_multithreaded(const uchar4 *h_rgbaImage, unsigned char *h_greyImage, int width, int height) {
    const int numThreads = 16;
    std::vector<std::thread> threads(numThreads);

    int chunkSize = height / numThreads;
    for (int i = 0; i < numThreads; ++i) {
        int startY = i * chunkSize;
        int endY = (i == numThreads - 1) ? height : (i + 1) * chunkSize;
        threads[i] = std::thread(rgba_to_greyscale_parallel, h_rgbaImage, h_greyImage, width, height, startY, endY);
    }

    for (auto &t : threads) {
        t.join();
    }
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


 
    // 将数据从主机内存复制到设备内存
    size_t numBytes = width * height * sizeof(uchar4);

    // 分配主机内存
    unsigned char* h_greyImage = (unsigned char*)malloc(numBytes);

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    // 执行 1000 次
    for (int i = 0; i < 1000; ++i) {
        rgba_to_greyscale_multithreaded(h_rgbaImage, h_greyImage, width, height);
    }

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();

    // 计算经过的时间
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " ms\n";


    // 保存图像为 PNG
    saveGreyImageToPNG(h_greyImage, "output.png", width, height);

    // 完成后，释放内存
    free(h_greyImage);
    return 0;
}
