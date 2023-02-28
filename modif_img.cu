#include "FreeImage.h"
#include <ctype.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 1920
#define HEIGHT 1024
#define BPP 24 // Since we're outputting three 8 bit RGB values

using namespace std;

char *output = (char *)"./output/new_img.png";

dim3 dimBlock;
dim3 dimGrid;

//! Diviser la matrice en 3 sous matrice pour le cache ( au lieu de faire +1 ,
//! +2 ) Passer de unsigned int à char ( vus que pixels )

__global__ void sobel(unsigned *d_img, unsigned *d_tmp, unsigned width,
                      unsigned height, unsigned level) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {
    int idx = ((y * width) + x) * 3;

    // If it's on the borders, put them on 0
    if (y == 0 || x == 0 || y == height - 1 || x == width - 1) {
      d_img[idx] = 0;
      d_img[idx + 1] = 0;
      d_img[idx + 2] = 0;
    } else {

      // Find the 8 neighbors
      int idv1 = (y * width + (x - 1) * 3);
      int idv2 = (y * width + (x + 1) * 3);
      int idv3 = (((y - 1) * width + x) * 3);
      int idv4 = (((y + 1) * width + x) * 3);
      int idv5 = (((y - 1) * width) + (x - 1)) * 3;
      int idv6 = (((y - 1) * width) + (x + 1)) * 3;
      int idv7 = (((y + 1) * width) + (x - 1)) * 3;
      int idv8 = (((y + 1) * width) + (x + 1)) * 3;

      // Gx convolution matrix
      int gx = -d_tmp[idv6] - d_tmp[idv5] - 2 * d_tmp[idv3] + d_tmp[idv8] +
               d_tmp[idv7] + 2 * d_tmp[idv4];

      // Gy convolution matrix
      int gy = -d_tmp[idv6] - d_tmp[idv8] - 2 * d_tmp[idv2] + d_tmp[idv5] +
               d_tmp[idv7] + 2 * d_tmp[idv1];

      // Gn = sqrt(gx*gx+gy*gy) < 100 -> (gx*gx+gy*gy) < 1000
      // Level is the difference between black and white
      int gn = (gx * gx + gy * gy) / 10000 * (level);

      // Replace pixels
      d_img[idx] = gn;
      d_img[idx + 1] = gn;
      d_img[idx + 2] = gn;
    }
  }
}

__global__ void canny(unsigned *d_img, unsigned *d_tmp, unsigned width,
                      unsigned height, unsigned level) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {
    int idx = ((y * width) + x) * 3;

    // If it's on the borders, put them on 0
    if (y == 0 || x == 0 || y == height - 1 || x == width - 1) {
      d_img[idx] = 0;
      d_img[idx + 1] = 0;
      d_img[idx + 2] = 0;
    } else {

      // Find the indexes of the 4 neighbors
      int idv1 = (y * width + (x - 1) * 3);
      int idv2 = (y * width + (x + 1) * 3);
      int idv3 = (((y - 1) * width + x) * 3);
      int idv4 = (((y + 1) * width + x) * 3);

      // Gx convolution matrix
      int gx = -d_tmp[idv1] + d_tmp[idv2];

      // Gy convolution matrix
      int gy = -d_tmp[idv3] + d_tmp[idv4];

      // Level is the difference between black and white
      int gn = sqrtf(gx * gx + gy * gy) * level;

      // Replace pixels
      d_img[idx] = gn;
      d_img[idx + 1] = gn;
      d_img[idx + 2] = gn;
    }
  }
}

__global__ void resize(unsigned *d_img, unsigned *img_new, unsigned width,
                       unsigned height, unsigned newWidth, unsigned newHeight) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < newHeight && x < newWidth) {
    int idx = ((y * newWidth) + x) * 3;

    // Find value on old width and height
    float u = (float)x / (float)newWidth * (float)width;
    float v = (float)y / (float)newHeight * (float)height;

    // Find x and y for neighbors
    int x1 = (int)u;
    int y1 = (int)v;
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    // Replace x and y if > width/height so it will not seg fault later
    if (x2 >= width) {
      x2 = width - 1;
    }
    if (y2 >= height) {
      y2 = height - 1;
    }

    // bilinear interpolation coefficients
    float a = u - (float)x1;
    float b = v - (float)y1;

    // Indexes of the 4 neighbors
    int idv1 = (((y1 * width) + x1) * 3);
    int idv2 = (((y1 * width) + x2) * 3);
    int idv3 = (((y2 * width) + x1) * 3);
    int idv4 = (((y2 * width) + x2) * 3);

    // Replace pixels
    img_new[idx] =
        (d_img[idv1] * (1.0f - a) * (1.0f - b) + d_img[idv2] * a * (1.0f - b) +
         d_img[idv3] * (1.0f - a) * b + d_img[idv4] * a * b);
    img_new[idx + 1] =
        (d_img[idv1 + 1] * (1.0f - a) * (1.0f - b) +
         d_img[idv2 + 1] * a * (1.0f - b) + d_img[idv3 + 1] * (1.0f - a) * b +
         d_img[idv4 + 1] * a * b);
    img_new[idx + 2] =
        (d_img[idv1 + 2] * (1.0f - a) * (1.0f - b) +
         d_img[idv2 + 2] * a * (1.0f - b) + d_img[idv3 + 2] * (1.0f - a) * b +
         d_img[idv4 + 2] * a * b);
  }
}

// ONLY + COLOR
// Put 0 to a color of a pixel

__global__ void onlyRed(unsigned *d_img, unsigned width, unsigned height) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {

    int idx = ((y * width) + x) * 3;

    d_img[idx + 1] = 0;
    d_img[idx + 2] = 0;
  }
}

__global__ void onlyBlue(unsigned *d_img, unsigned width, unsigned height) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {

    int idx = ((y * width) + x) * 3;

    d_img[idx] = 0;
    d_img[idx + 1] = 0;
  }
}

__global__ void onlyGreen(unsigned *d_img, unsigned width, unsigned height) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {

    int idx = ((y * width) + x) * 3;

    d_img[idx] = 0;
    d_img[idx + 2] = 0;
  }
}
__global__ void onlyGreenRed(unsigned *d_img, unsigned width, unsigned height) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {

    int idx = ((y * width) + x) * 3;
    d_img[idx + 2] = 0;
  }
}
__global__ void onlyGreenBlue(unsigned *d_img, unsigned width,
                              unsigned height) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {

    int idx = ((y * width) + x) * 3;
    d_img[idx] = 0;
  }
}

__global__ void onlyRedBlue(unsigned *d_img, unsigned width, unsigned height) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {

    int idx = ((y * width) + x) * 3;
    d_img[idx + 1] = 0;
  }
}

// Diapositive = c-color
__global__ void diapositive(unsigned *d_img, unsigned width, unsigned height) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {

    int idx = ((y * width) + x) * 3;

    d_img[idx] = 255 - d_img[idx];
    d_img[idx + 1] = 255 - d_img[idx + 1];
    d_img[idx + 2] = 255 - d_img[idx + 2];
  }
}

__global__ void blur(unsigned *d_img, unsigned *d_tmp, unsigned width,
                     unsigned height) {

  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {

    int count = 0, count1 = 0, count2 = 0, c = 0;

    // find neighbors
    int idx = ((y * width) + x) * 3;
    int idv1 = (((y + 1) * width) + x) * 3;
    int idv2 = (((y - 1) * width) + x) * 3;
    int idv3 = ((y * width) + (x + 1)) * 3;
    int idv4 = ((y * width) + (x - 1)) * 3;

    // Since it's -1 and +1 for the neigbors, the if are the exceptions for the
    // borders, so you don't count neighbors over the image
    if (x > 0) {
      count += d_tmp[idv3];
      count1 += d_tmp[idv3 + 1];
      count2 += d_tmp[idv3 + 2];
      c++;
    }

    if (x < width - 1) {
      count += d_tmp[idv4];
      count1 += d_tmp[idv4 + 1];
      count2 += d_tmp[idv4 + 2];
      c++;
    }

    if (y > 0) {
      count += d_tmp[idv1];
      count1 += d_tmp[idv1 + 1];
      count2 += d_tmp[idv1 + 2];
      c++;
    }

    if (y < height - 1) {
      count += d_tmp[idv2];
      count1 += d_tmp[idv2 + 1];
      count2 += d_tmp[idv2 + 2];
      c++;
    }

    // Sum of count for every color
    count += d_tmp[idx];
    count1 += d_tmp[idx + 1];
    count2 += d_tmp[idx + 2];

    c++;

    // Mean of the sums
    d_img[idx] = count / c;
    d_img[idx + 1] = count1 / c;
    d_img[idx + 2] = count2 / c;
  }
}

// SATURATION + COLORS
// Put 255/1.5 on a color
// For grey = weighted mean
__global__ void saturationGrey(unsigned *d_img, unsigned width,
                               unsigned height) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {

    int idx = ((y * width) + x) * 3;
    int grey =
        d_img[idx] * 0.299 + d_img[idx + 1] * 0.587 + d_img[idx + 2] * 0.114;

    d_img[idx] = grey;
    d_img[idx + 1] = grey;
    d_img[idx + 2] = grey;
  }
}

__global__ void saturationBlue(unsigned *d_img, unsigned width,
                               unsigned height) {

  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {

    int idx = ((y * width) + x) * 3;
    d_img[idx + 2] = 0xFF / 1.5;
  }
}

__global__ void saturationRed(unsigned *d_img, unsigned width,
                              unsigned height) {

  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {

    int idx = ((y * width) + x) * 3;
    d_img[idx] = 0xFF / 1.5;
  }
}

__global__ void saturationGreen(unsigned *d_img, unsigned width,
                                unsigned height) {

  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {
    int idx = ((y * width) + x) * 3;
    d_img[idx + 1] = 0xFF / 1.5;
  }
}

__global__ void saturationCyan(unsigned *d_img, unsigned width,
                               unsigned height) {

  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {
    int idx = ((y * width) + x) * 3;
    d_img[idx + 1] = 0xFF / 1.5;
    d_img[idx + 2] = 0xFF / 1.5;
  }
}

__global__ void saturationYellow(unsigned *d_img, unsigned width,
                                 unsigned height) {

  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {
    int idx = ((y * width) + x) * 3;
    d_img[idx] = 0xFF / 1.5;
    d_img[idx + 1] = 0xFF / 1.5;
  }
}
__global__ void saturationMagenta(unsigned *d_img, unsigned width,
                                  unsigned height) {

  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {
    int idx = ((y * width) + x) * 3;
    d_img[idx] = 0xFF / 1.5;
    d_img[idx + 2] = 0xFF / 1.5;
  }
}

// Horizontal symmetry
// just invert the top with the bottom
__global__ void symhorizontal(unsigned *d_img, unsigned *d_tmp, unsigned width,
                              unsigned height) {

  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width) {
    int ida = ((y * width) + x) * 3;
    int idb = ((width * height) - ((y * width) + x)) * 3;
    d_img[ida] = d_tmp[idb];
    d_img[ida + 1] = d_tmp[idb + 1];
    d_img[ida + 2] = d_tmp[idb + 2];
  }
}

// CPU FUNCTIONS

void saturation(unsigned *img, unsigned width, unsigned height, unsigned size,
                int color) {
  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid((height / 32) + 1, (width / 32) + 1, 1);

  unsigned int *d_img = (unsigned int *)malloc(size);

  memcpy(d_img, img, size);

  unsigned *d_a;

  cudaMalloc((void **)&d_a, size);

  cudaMemcpy(d_a, d_img, size, cudaMemcpyHostToDevice);

  // Switch for all the colors
  switch (color) {
  case 0: {
    saturationRed<<<dimGrid, dimBlock>>>(d_a, width, height);
    printf("Saturation RED:\n\tDone successfully\n");
    break;
  }
  case 1: {
    saturationBlue<<<dimGrid, dimBlock>>>(d_a, width, height);
    printf("Saturation BLUE:\n\tDone successfully\n");
    break;
  }
  case 2: {
    saturationGreen<<<dimGrid, dimBlock>>>(d_a, width, height);
    printf("Saturation GREEN:\n\tDone successfully\n");
    break;
  }
  case 3: {
    saturationCyan<<<dimGrid, dimBlock>>>(d_a, width, height);
    printf("Saturation CYAN:\n\tDone successfully\n");
    break;
  }
  case 4: {
    saturationYellow<<<dimGrid, dimBlock>>>(d_a, width, height);
    printf("Saturation YELLOW:\n\tDone successfully\n");
    break;
  }
  case 5: {
    saturationGrey<<<dimGrid, dimBlock>>>(d_a, width, height);
    printf("Saturation GREY:\n\tDone successfully\n");
    break;
  }
  case 6: {
    saturationMagenta<<<dimGrid, dimBlock>>>(d_a, width, height);
    printf("Saturation MAGENTA:\n\tDone successfully\n");
    break;
  }
  }

  cudaMemcpy(img, d_a, size, cudaMemcpyDeviceToHost);
}

void popart(unsigned *img, unsigned width, unsigned height, unsigned pitch,
            FIBITMAP *bitmap) {

  printf("POP ART:\n");

  // Create rescaled image (1/4 of the base one)
  FIBITMAP *split =
      FreeImage_Rescale(bitmap, width / 2, height / 2, FILTER_BOX);

  unsigned widthSplt = FreeImage_GetWidth(split);
  unsigned heightSplt = FreeImage_GetHeight(split);
  unsigned pitchSplt = FreeImage_GetPitch(split);

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid((heightSplt / 32) + 1, (widthSplt / 32) + 1, 1);

  printf("\tProcessing sub-Image of size %d x %d\n", widthSplt, heightSplt);

  int sizeSplt = sizeof(unsigned int) * 3 * widthSplt * heightSplt;

  unsigned int *imgSplt = (unsigned int *)malloc(sizeSplt);
  unsigned int *d_tmpSplt = (unsigned int *)malloc(sizeSplt);

  // Copy pixel of the split image
  BYTE *bitsSplt = (BYTE *)FreeImage_GetBits(split);
  for (int y = 0; y < heightSplt; y++) {
    BYTE *pixelSplt = (BYTE *)bitsSplt;
    for (int x = 0; x < widthSplt; x++) {
      int idx = ((y * widthSplt) + x) * 3;
      imgSplt[idx] = pixelSplt[FI_RGBA_RED];
      imgSplt[idx + 1] = pixelSplt[FI_RGBA_GREEN];
      imgSplt[idx + 2] = pixelSplt[FI_RGBA_BLUE];
      pixelSplt += 3;
    }
    // next line
    bitsSplt += pitchSplt;
  }

  unsigned int *d_imgSplt;
  unsigned int *d_imgSplt2;
  unsigned int *d_imgSplt3;
  unsigned int *d_imgSplt4;

  // Pin memory
  cudaMallocHost((void **)&d_imgSplt2, sizeSplt);
  cudaMallocHost((void **)&d_imgSplt3, sizeSplt);
  cudaMallocHost((void **)&d_imgSplt4, sizeSplt);
  cudaMallocHost((void **)&d_imgSplt, sizeSplt);

  // Get copy of the split image
  memcpy(d_imgSplt, imgSplt, sizeSplt);
  memcpy(d_imgSplt2, imgSplt, sizeSplt);
  memcpy(d_imgSplt3, imgSplt, sizeSplt);
  memcpy(d_imgSplt4, imgSplt, sizeSplt);

  unsigned *d1, *d2, *d3, *d4;

  // Prepare copy for the devices
  cudaMalloc((void **)&d1, sizeSplt);
  cudaMalloc((void **)&d2, sizeSplt);
  cudaMalloc((void **)&d3, sizeSplt);
  cudaMalloc((void **)&d4, sizeSplt);

  cudaStream_t stream[4];

  // Create streams
  cudaStreamCreate(&stream[0]);
  cudaStreamCreate(&stream[1]);
  cudaStreamCreate(&stream[2]);
  cudaStreamCreate(&stream[3]);

  // Sync copy to device
  cudaMemcpyAsync(d1, d_imgSplt, sizeSplt, cudaMemcpyHostToDevice, stream[0]);
  cudaMemcpyAsync(d2, d_imgSplt2, sizeSplt, cudaMemcpyHostToDevice, stream[1]);
  cudaMemcpyAsync(d3, d_imgSplt3, sizeSplt, cudaMemcpyHostToDevice, stream[2]);
  cudaMemcpyAsync(d4, d_imgSplt4, sizeSplt, cudaMemcpyHostToDevice, stream[3]);

  // Kernel function for all the saturations
  saturationRed<<<dimGrid, dimBlock, 0, stream[0]>>>(d1, widthSplt, heightSplt);
  saturationGrey<<<dimGrid, dimBlock, 0, stream[1]>>>(d2, widthSplt,
                                                      heightSplt);
  saturationBlue<<<dimGrid, dimBlock, 0, stream[2]>>>(d3, widthSplt,
                                                      heightSplt);
  saturationGreen<<<dimGrid, dimBlock, 0, stream[3]>>>(d4, widthSplt,
                                                       heightSplt);

  // Async cop to host
  cudaMemcpyAsync(d_imgSplt, d1, sizeSplt, cudaMemcpyDeviceToHost, stream[0]);
  cudaMemcpyAsync(d_imgSplt2, d2, sizeSplt, cudaMemcpyDeviceToHost, stream[1]);
  cudaMemcpyAsync(d_imgSplt3, d3, sizeSplt, cudaMemcpyDeviceToHost, stream[2]);
  cudaMemcpyAsync(d_imgSplt4, d4, sizeSplt, cudaMemcpyDeviceToHost, stream[3]);

  // Synchronize streams
  cudaStreamSynchronize(stream[0]);
  cudaStreamSynchronize(stream[1]);
  cudaStreamSynchronize(stream[2]);
  cudaStreamSynchronize(stream[3]);

  // Copy back on pixels to save the image
  bitsSplt = (BYTE *)FreeImage_GetBits(bitmap);
  for (int y = 0; y < heightSplt; y++) {
    BYTE *pixel = (BYTE *)bitsSplt;
    for (int x = 0; x < widthSplt; x++) {
      RGBQUAD newcolor;

      int idx = ((y * widthSplt) + x) * 3;
      newcolor.rgbRed = d_imgSplt[idx];
      newcolor.rgbGreen = d_imgSplt[idx + 1];
      newcolor.rgbBlue = d_imgSplt[idx + 2];

      if (!FreeImage_SetPixelColor(bitmap, x, y, &newcolor)) {
        fprintf(stderr, "(%d, %d) Fail...\n", x, y);
      }

      pixel += 3;
    }
    // next line
    bitsSplt += pitch;
  }

  bitsSplt = (BYTE *)FreeImage_GetBits(bitmap);

  for (int y = 0; y < heightSplt; y++) {
    BYTE *pixel = (BYTE *)bitsSplt;
    for (int x = 0; x < widthSplt; x++) {
      RGBQUAD newcolor;

      int idx = ((y * widthSplt) + x) * 3;
      newcolor.rgbRed = d_imgSplt2[idx];
      newcolor.rgbGreen = d_imgSplt2[idx + 1];
      newcolor.rgbBlue = d_imgSplt2[idx + 2];

      if (!FreeImage_SetPixelColor(bitmap, x + widthSplt, y + heightSplt,
                                   &newcolor)) {
        fprintf(stderr, "(%d, %d) Fail...\n", x, y);
      }

      pixel += 3;
    }
    // next line
    bitsSplt += pitchSplt;
  }
  bitsSplt = (BYTE *)FreeImage_GetBits(bitmap);

  for (int y = 0; y < heightSplt; y++) {
    BYTE *pixel = (BYTE *)bitsSplt;
    for (int x = 0; x < widthSplt; x++) {
      RGBQUAD newcolor;

      int idx = ((y * widthSplt) + x) * 3;
      newcolor.rgbRed = d_imgSplt3[idx];
      newcolor.rgbGreen = d_imgSplt3[idx + 1];
      newcolor.rgbBlue = d_imgSplt3[idx + 2];

      if (!FreeImage_SetPixelColor(bitmap, x, y + heightSplt, &newcolor)) {
        fprintf(stderr, "(%d, %d) Fail...\n", x, y);
      }

      pixel += 3;
    }
    // next line
    bitsSplt += pitchSplt;
  }
  bitsSplt = (BYTE *)FreeImage_GetBits(bitmap);

  for (int y = 0; y < heightSplt; y++) {
    BYTE *pixel = (BYTE *)bitsSplt;
    for (int x = 0; x < widthSplt; x++) {
      RGBQUAD newcolor;

      int idx = ((y * widthSplt) + x) * 3;
      newcolor.rgbRed = d_imgSplt4[idx];
      newcolor.rgbGreen = d_imgSplt4[idx + 1];
      newcolor.rgbBlue = d_imgSplt4[idx + 2];

      if (!FreeImage_SetPixelColor(bitmap, x + widthSplt, y, &newcolor)) {
        fprintf(stderr, "(%d, %d) Fail...\n", x, y);
      }

      pixel += 3;
    }
    // next line
    bitsSplt += pitchSplt;
  }

  // Save image
  if (FreeImage_Save(FIF_PNG, bitmap, output, 0))
    printf("\tImage saved at %s\n", output);
}

void canny_cpu(unsigned *img, unsigned width, unsigned height, unsigned level,
               unsigned size) {

  printf("Canny:\n\tStarting with level %u\n", level);
  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid((height / 32) + 1, (width / 32) + 1, 1);

  unsigned int *d_img = (unsigned int *)malloc(size);

  memcpy(d_img, img, size);

  unsigned *d_a, *d_b;

  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);

  cudaMemcpy(d_a, d_img, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, d_img, size, cudaMemcpyHostToDevice);

  canny<<<dimGrid, dimBlock>>>(d_a, d_b, width, height, level);

  cudaMemcpy(img, d_a, size, cudaMemcpyDeviceToHost);

  printf("\tDone successfully\n");
}

void sobel_cpu(unsigned *img, unsigned width, unsigned height, unsigned level,
               unsigned size) {
  printf("Sobel:\n\tStarting with level %u\n", level);

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid((height / 32) + 1, (width / 32) + 1, 1);

  unsigned int *d_img = (unsigned int *)malloc(size);

  memcpy(d_img, img, size);

  unsigned *d_a, *d_b;

  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);

  cudaMemcpy(d_a, d_img, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, d_img, size, cudaMemcpyHostToDevice);

  sobel<<<dimGrid, dimBlock>>>(d_a, d_b, width, height, level);

  cudaMemcpy(img, d_a, size, cudaMemcpyDeviceToHost);
  printf("\tDone successfully\n");
}

void blur_cpu(unsigned *img, unsigned width, unsigned height, unsigned level,
              unsigned size) {
  printf("Blur:\n\tStarting with level %u\n", level);

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid((height / 32) + 1, (width / 32) + 1, 1);

  unsigned int *d_img = (unsigned int *)malloc(size);

  memcpy(d_img, img, size);

  unsigned *d_a, *d_b;

  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);

  cudaMemcpy(d_a, d_img, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, d_img, size, cudaMemcpyHostToDevice);

  for (int i = 0; i < level; i++) {
    blur<<<dimGrid, dimBlock>>>(d_a, d_b, width, height);
    cudaMemcpy(d_b, d_a, size, cudaMemcpyDeviceToDevice);
  }

  cudaMemcpy(d_img, d_a, size, cudaMemcpyDeviceToHost);
  memcpy(img, d_img, size);
  printf("\tDone successfully\n");
}

void resize_cpu(unsigned *img, unsigned *width, unsigned *height, double factor,
                unsigned *size) {
  printf("Resize:\n\tStarting with factor %u\n", factor);

  int newHeight = *height * factor;
  int newWidth = *width * factor;

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid((newHeight / 32) + 1, (newWidth / 32) + 1, 1);

  unsigned int *d_img = (unsigned int *)malloc(*size);

  memcpy(d_img, img, *size);

  unsigned *d_a, *d_b;

  cudaMalloc((void **)&d_a, *size);
  cudaMalloc((void **)&d_b, *size);

  cudaMemcpy(d_a, d_img, *size, cudaMemcpyHostToDevice);

  resize<<<dimGrid, dimBlock>>>(d_a, d_b, *width, *height, newWidth, newHeight);

  *size = 3 * newWidth * newHeight * sizeof(unsigned int);

  cudaMemcpy(d_img, d_b, *size, cudaMemcpyDeviceToHost);

  *width = newWidth;

  *height = newHeight;

  memcpy(img, d_img, *size);

  printf("\tDone successfully\n");
}

void only(unsigned *img, unsigned width, unsigned height, unsigned size,
          int color) {
  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid((height / 32) + 1, (width / 32) + 1, 1);

  unsigned int *d_img = (unsigned int *)malloc(size);

  memcpy(d_img, img, size);

  unsigned *d_a;

  cudaMalloc((void **)&d_a, size);

  cudaMemcpy(d_a, d_img, size, cudaMemcpyHostToDevice);

  // switch for colors
  switch (color) {
  case 0: {
    onlyRed<<<dimGrid, dimBlock>>>(d_a, width, height);
    printf("Only RED:\n\tDone successfully\n");
    break;
  }
  case 1: {
    onlyBlue<<<dimGrid, dimBlock>>>(d_a, width, height);
    printf("Only BLUE:\n\tDone successfully\n");
    break;
  }
  case 2: {
    onlyGreen<<<dimGrid, dimBlock>>>(d_a, width, height);
    printf("Only GREEN:\n\tDone successfully\n");
    break;
  }
  case 3: {
    onlyGreenRed<<<dimGrid, dimBlock>>>(d_a, width, height);
    printf("Only GREEN-RED:\n\tDone successfully\n");
    break;
  }
  case 4: {
    onlyGreenBlue<<<dimGrid, dimBlock>>>(d_a, width, height);
    printf("Only GREEN-BLUE:\n\tDone successfully\n");
    break;
  }
  case 5: {
    onlyRedBlue<<<dimGrid, dimBlock>>>(d_a, width, height);
    printf("Only RED-BLUE:\n\tDone successfully\n");
    break;
  }
  }

  cudaMemcpy(img, d_a, size, cudaMemcpyDeviceToHost);
}

void diapositive_cpu(unsigned *img, unsigned width, unsigned height,
                     unsigned size) {
  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid((height / 32) + 1, (width / 32) + 1, 1);

  unsigned int *d_img = (unsigned int *)malloc(size);

  memcpy(d_img, img, size);

  unsigned *d_a;

  cudaMalloc((void **)&d_a, size);

  cudaMemcpy(d_a, d_img, size, cudaMemcpyHostToDevice);

  diapositive<<<dimGrid, dimBlock>>>(d_a, width, height);

  cudaMemcpy(d_img, d_a, size, cudaMemcpyDeviceToHost);

  memcpy(img, d_img, size);

  printf("Diapositive:\n\tDone successfully\n");
}
void hor_sym(unsigned *img, unsigned width, unsigned height, unsigned size) {
  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid((height / 32) + 1, (width / 32) + 1, 1);

  unsigned int *d_img = (unsigned int *)malloc(size);

  memcpy(d_img, img, size);

  unsigned *d_a, *d_b;

  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);

  cudaMemcpy(d_a, d_img, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, d_img, size, cudaMemcpyHostToDevice);

  symhorizontal<<<dimGrid, dimBlock>>>(d_a, d_b, width, height);

  cudaMemcpy(img, d_a, size, cudaMemcpyDeviceToHost);
  printf("Horizontal symmetry:\n\tDone successfully\n");
}

int main(int argc, char **argv) {
  FreeImage_Initialise();
  char *input = (char *)"./input/img.jpg";

  // load and decode default file
  FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(input);

  FIBITMAP *bitmap = FreeImage_Load(FIF_JPEG, input, 0);

  if (!bitmap)
    exit(1); // WTF?! We can't even allocate images ? Die !

  unsigned width = FreeImage_GetWidth(bitmap);
  unsigned height = FreeImage_GetHeight(bitmap);
  unsigned pitch = FreeImage_GetPitch(bitmap);

  printf("Load default file from %s, width: %u height: %u pitch: %u\n", input,
         width, height, pitch);

  unsigned size = sizeof(unsigned int) * 3 * width * height;

  unsigned int *img = (unsigned int *)malloc(size);

  // Load pixels
  BYTE *bits = (BYTE *)FreeImage_GetBits(bitmap);
  for (int y = 0; y < height; y++) {
    BYTE *pixel = (BYTE *)bits;
    for (int x = 0; x < width; x++) {
      int idx = ((y * width) + x) * 3;
      img[idx] = pixel[FI_RGBA_RED];
      img[idx + 1] = pixel[FI_RGBA_GREEN];
      img[idx + 2] = pixel[FI_RGBA_BLUE];
      pixel += 3;
    }
    // next line
    bits += pitch;
  }

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--saturation") == 0 && i + 1 < argc) {
      if (strcmp(argv[i + 1], "RED") == 0) {
        saturation(img, width, height, size, 0);
      } else if (strcmp(argv[i + 1], "GREEN") == 0) {
        saturation(img, width, height, size, 2);
      } else if (strcmp(argv[i + 1], "BLUE") == 0) {
        saturation(img, width, height, size, 1);
      } else if (strcmp(argv[i + 1], "CYAN") == 0) {
        saturation(img, width, height, size, 3);
      } else if (strcmp(argv[i + 1], "YELLOW") == 0) {
        saturation(img, width, height, size, 4);
      } else if (strcmp(argv[i + 1], "GREY") == 0) {
        saturation(img, width, height, size, 5);
      } else if (strcmp(argv[i + 1], "MAGENTA") == 0) {
        saturation(img, width, height, size, 6);
      }
    } else if (strcmp(argv[i], "--blur") == 0) {
      if (i + 1 < argc) {
        int level;
        if (sscanf(argv[i + 1], "%d", &level) != 1)
          blur_cpu(img, width, height, 50, size);
        else
          blur_cpu(img, width, height, level, size);
      } else {
        // Default = 20;
        blur_cpu(img, width, height, 50, size);
      }
    } else if (strcmp(argv[i], "--sobel") == 0) {
      if (i + 1 < argc) {
        int level;
        if (sscanf(argv[i + 1], "%d", &level) != 1)
          sobel_cpu(img, width, height, 5000, size);
        else
          sobel_cpu(img, width, height, level, size);
      } else
        sobel_cpu(img, width, height, 5000, size);

    } else if (strcmp(argv[i], "--output") == 0) {
      if (i + 1 < argc) {
        output = (char *)malloc(sizeof(char) * strlen(argv[i + 1]));
        sprintf(output, "./output/%s", argv[i + 1]);
      }
    } else if (strcmp(argv[i], "--input") == 0) {
      if (i + 1 < argc) {
        input = (char *)malloc(sizeof(char) * strlen(argv[i + 1]));
        sprintf(input, "./input/%s", argv[i + 1]);

        // load and decode custom file
        FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(input);

        FIBITMAP *bitmap = FreeImage_Load(FIF_JPEG, input, 0);

        if (!bitmap)
          exit(1); // WTF?! We can't even allocate images ? Die !

        unsigned width = FreeImage_GetWidth(bitmap);
        unsigned height = FreeImage_GetHeight(bitmap);
        unsigned pitch = FreeImage_GetPitch(bitmap);

        printf("Load custom file from %s, width: %u height: %u pitch: %u\n",
               input, width, height, pitch);

        unsigned size = sizeof(unsigned int) * 3 * width * height;

        unsigned int *img = (unsigned int *)malloc(size);

        BYTE *bits = (BYTE *)FreeImage_GetBits(bitmap);
        for (int y = 0; y < height; y++) {
          BYTE *pixel = (BYTE *)bits;
          for (int x = 0; x < width; x++) {
            int idx = ((y * width) + x) * 3;
            img[idx] = pixel[FI_RGBA_RED];
            img[idx + 1] = pixel[FI_RGBA_GREEN];
            img[idx + 2] = pixel[FI_RGBA_BLUE];
            pixel += 3;
          }
          // next line
          bits += pitch;
        }
      }
    } else if (strcmp(argv[i], "--canny") == 0) {
      if (i + 1 < argc) {
        int level;
        if (sscanf(argv[i + 1], "%d", &level) != 1)
          canny_cpu(img, width, height, 1, size);
        else
          canny_cpu(img, width, height, level, size);
      } else
        // Default = 1;
        canny_cpu(img, width, height, 1, size);

    } else if (strcmp(argv[i], "--popart") == 0) {
      popart(img, width, height, pitch, bitmap);
      return 0;
    } else if (strcmp(argv[i], "--diapositive") == 0) {
      diapositive_cpu(img, width, height, size);
    } else if (strcmp(argv[i], "--sym") == 0) {
      hor_sym(img, width, height, size);
    } else if (strcmp(argv[i], "--resize") == 0) {
      if ((i + 1 < argc)) {
        char *endptr;
        double factor = (double)strtod(argv[i + 1], &endptr);
        if ((endptr == argv[i + 1]) || (*endptr != '\0'))
          resize_cpu(img, &width, &height, (double)0.5, &size);
        else
          resize_cpu(img, &width, &height, factor, &size);

      } else
        // Default = 0.5;
        resize_cpu(img, &width, &height, (double)0.5, &size);

    } else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
      if (strcmp(argv[i + 1], "RED") == 0) {
        only(img, width, height, size, 0);
      } else if (strcmp(argv[i + 1], "GREEN") == 0) {
        only(img, width, height, size, 2);
      } else if (strcmp(argv[i + 1], "BLUE") == 0) {
        only(img, width, height, size, 1);
      } else if (strcmp(argv[i + 1], "GREENRED") == 0) {
        only(img, width, height, size, 3);
      } else if (strcmp(argv[i + 1], "GREENBLUE") == 0) {
        only(img, width, height, size, 4);
      } else if (strcmp(argv[i + 1], "REDBLUE") == 0) {
        only(img, width, height, size, 5);
      }
    } else if (strcmp(argv[i], "--help") == 0) {
      printf(
          "- \033[1m\033[38;5;93msobel\033[0m <\033[38;5;57mlevel\033[0m> | "
          "Apply Sobel filter for edge "
          "detection\n\tLevel: Integer number to define black to white in "
          "the result\n\tDefault: \033[38;5;28m5000\033[0m\n\n- "
          "\033[1m\033[38;5;93mcanny\033[0m "
          "<\033[38;5;57mlevel\033[0m> "
          "| "
          "Apply Canny "
          "filter for edge detection\n\tLevel: Integer number to define "
          "black to white in the result. \n\tDefault: "
          "\033[38;5;28m1\033[0m\n\n- "
          "\033[1m\033[38;5;93mpopart\033[0m | Apply "
          "a pop-art filter with 4 same images with 4 differents saturation "
          "filter (blue/red/green/grey)\n\n- \033[1m\033[38;5;93mresize\033[0m "
          "<\033[38;5;57mfactor\033[0m> "
          "| Resize "
          "image\n\tfactor: Double number for the resizing factor, it will "
          "be (width/height*factor), to divide the size, use > 1 numbers, "
          "ie: size/2 = 0.5. \n\tDefault : \033[38;5;28m0.5\033[0m\n\n- "
          "\033[1m\033[38;5;93monly\033[0m "
          "<\033[38;5;57mcolor\033[0m> "
          "| Keep only "
          "one colour on the pixels\n\tColors: "
          "\e[0;91mRED/\e[0;94mBLUE/\e[0;92mGREEN/\033[38;5;36mGREENBLUE/"
          "\033[38;5;41mGREENRED/\033[38;5;35m"
          "REDBLUE\n\n- \033[1m\033[38;5;93msaturation\033[0m "
          "<\033[38;5;57mcolor\033[0m> | Saturate a colour on the "
          "pixels\n\tColors: "
          "\e[0;91mRED/\e[0;94mBLUE/\e[0;92mGREEN/\e[0;96mCYAN/\e[0;61mGREY/"
          "\e[0;93mYELLOW/"
          "\e[0;95mMAGENTA\n\n- \033[1m\033[38;5;93msym\033[0m | Apply an "
          "horizontal symmetry\n\n- \033[1m\033[38;5;93mblur\033[0m "
          "<\033[38;5;57miterations\033[0m> | "
          "Apply "
          "a blur "
          "effect\n\tIterations: Integer number, how many time the blur "
          "effect will be done, at minimum 20 iterations is "
          "recommended\n\tDefault: \033[38;5;28m20\033[0m  \n\n- "
          "\033[1m\033[38;5;93moutput\033[0m "
          "<\033[38;5;57mname\033[0m> | "
          "Custom output "
          "image name\n\tName: String value, will still be stored in the "
          "./output/ directory\n\tDefault: "
          "\033[38;5;28mnew_img.png\033[0m\n\n- "
          "\033[1m\033[38;5;93minput\033[0m <\033[38;5;57mname\033[0m> | "
          "Custom input image name\n\tName: String value, will still be "
          "retrieved from the ./input/ directory\n\tDefault: "
          "\033[38;5;28mimg.png\033[0m\n\n- "
          "\033[1m\033[38;5;93mhelp\033[0m | Prints this message and exits "
          "program\033[0m\n\n\n\033[38;5;1m(!)\033[0m The order of the "
          "arguments is necessary, "
          "the actions will be executed in order. If for example, you perform "
          "a filter before modifying the input file, then the filter will be "
          "done on the base image.\n\033[38;5;117mExample: ./modif_cuda "
          "--saturation RED\033[0m\n");
      return 0;
    }
  }

  unsigned *save = (unsigned *)malloc(size);
  memcpy(save, img, size);

  printf("\tSaving on image height: %u width: %u\n", height, width);

  FIBITMAP *newImage = FreeImage_Allocate(width, height, 24);

  // Save pixels
  bits = (BYTE *)FreeImage_GetBits(newImage);
  for (int y = 0; y < height; y++) {
    BYTE *pixel = (BYTE *)bits;
    for (int x = 0; x < width; x++) {
      RGBQUAD newcolor;

      int idx = ((y * width) + x) * 3;
      newcolor.rgbRed = save[idx];
      newcolor.rgbGreen = save[idx + 1];
      newcolor.rgbBlue = save[idx + 2];

      if (!FreeImage_SetPixelColor(newImage, x, y, &newcolor)) {
        fprintf(stderr, "(%d, %d) Fail...\n", x, y);
      }

      pixel += 3;
    }
    // next line
    bits += pitch;
  }

  if (FreeImage_Save(FIF_PNG, newImage, output, 0))
    printf("\tImage saved at %s\n", output);

  FreeImage_DeInitialise(); // Cleanup !
}
