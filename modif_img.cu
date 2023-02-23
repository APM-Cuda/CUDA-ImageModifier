#include <iostream>
#include <string.h>
#include "FreeImage.h"
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 1920
#define HEIGHT 1024
#define BPP 24 // Since we're outputting three 8 bit RGB values

using namespace std;

//! Diviser la matrice en 3 sous matrice pour le cache ( au lieu de faire +1 , +2 )
//! Passer de unsigned int à char ( vus que pixels )

__global__ void sobel(unsigned *d_img, unsigned *d_tmp, unsigned width, unsigned height)
{
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width)
  {
    int idx = ((y * width) + x) * 3;

    if (y == 0 || x == 0 || y == height - 1 || x == width - 1)
    {
      d_img[idx] = 0;
      d_img[idx + 1] = 0;
      d_img[idx + 2] = 0;
    }
    else
    {

      int idv1 = (y * width + (x - 1) * 3);
      int idv2 = (y * width + (x + 1) * 3);
      int idv3 = (((y - 1) * width + x) * 3);
      int idv4 = (((y + 1) * width + x) * 3);
      int idv5 = (((y - 1) * width) + (x - 1)) * 3;
      int idv6 = (((y - 1) * width) + (x + 1)) * 3;
      int idv7 = (((y + 1) * width) + (x - 1)) * 3;
      int idv8 = (((y + 1) * width) + (x + 1)) * 3;

      int gx = -d_tmp[idv6] - d_tmp[idv5] - 2 * d_tmp[idv3] + d_tmp[idv8] + d_tmp[idv7] + 2 * d_tmp[idv4];

      int gy = -d_tmp[idv6] - d_tmp[idv8] - 2 * d_tmp[idv2] + d_tmp[idv5] + d_tmp[idv7] + 2 * d_tmp[idv1];

      int gn = sqrtf(gx * gx + gy * gy);

      d_img[idx + 0] = gn;

      d_img[idx + 1] = gn;

      d_img[idx + 2] = gn;
    }
  }
}

__global__ void gris(unsigned *d_img, unsigned width, unsigned height)
{
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width)
  {

    int idx = ((y * width) + x) * 3;
    int grey = d_img[idx + 0] * 0.299 + d_img[idx + 1] * 0.587 + d_img[idx + 2] * 0.114;

    d_img[idx + 0] = grey;
    d_img[idx + 1] = grey;
    d_img[idx + 2] = grey;
    // printf("%d %d %f\n",y,x, count);
  }
}
__global__ void flou(unsigned *d_img, unsigned width, unsigned height)
{

  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width)
  {

    int count = 0, count1 = 0, count2 = 0, c = 0;
    int idx = ((y * width) + x) * 3;
    int idv1 = (((y + 1) * width) + x) * 3;
    int idv2 = (((y - 1) * width) + x) * 3;
    int idv3 = ((y * width) + (x + 1)) * 3;
    int idv4 = ((y * width) + (x - 1)) * 3;

    if (x > 0)
    {
      count += d_img[idv3];
      count1 += d_img[idv3 + 1];
      count2 += d_img[idv3 + 2];
      c++;
    }

    if (x < width - 1)
    {
      count += d_img[idv4];
      count1 += d_img[idv4 + 1];
      count2 += d_img[idv4 + 2];
      c++;
    }

    if (y > 0)
    {
      count += d_img[idv1];
      count1 += d_img[idv1 + 1];
      count2 += d_img[idv1 + 2];
      c++;
    }

    if (y < height - 1)
    {
      count += d_img[idv2];
      count1 += d_img[idv2 + 1];
      count2 += d_img[idv2 + 2];
      c++;
    }

    count += d_img[idx];
    count1 += d_img[idx + 1];
    count2 += d_img[idx + 2];

    c++;

    d_img[idx + 0] = count / c;
    d_img[idx + 1] = count1 / c;
    d_img[idx + 2] = count2 / c;
  }
}

__global__ void saturation(unsigned *d_img, unsigned width, unsigned height)
{

  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width)
  {

    int idx = ((y * width) + x) * 3;
    d_img[idx + 2] = 0xFF / 1.5;
  }
}

__global__ void symhorizontal(unsigned *d_img, unsigned *d_tmp, unsigned width, unsigned height)
{

  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < height && x < width)
  {
    int ida = ((y * width) + x) * 3;
    int idb = ((width * height) - ((y * width) + x)) * 3;
    d_img[ida + 0] = d_tmp[idb + 0];
    d_img[ida + 1] = d_tmp[idb + 1];
    d_img[ida + 2] = d_tmp[idb + 2];
  }
}

int main(int argc, char **argv)
{
  FreeImage_Initialise();
  const char *PathName = "img.jpg";
  const char *PathDest = "new_img.png";
  // load and decode a regular file
  FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(PathName);

  FIBITMAP *bitmap = FreeImage_Load(FIF_JPEG, PathName, 0);

  if (!bitmap)
    exit(1); // WTF?! We can't even allocate images ? Die !

  unsigned width = FreeImage_GetWidth(bitmap);
  unsigned height = FreeImage_GetHeight(bitmap);
  unsigned pitch = FreeImage_GetPitch(bitmap);

  printf("Processing Image of size %d x %d\n", width, height);

  int size = sizeof(unsigned int) * 3 * width * height;

  unsigned int *img = (unsigned int *)malloc(size);
  unsigned int *d_img = (unsigned int *)malloc(size);
  unsigned int *d_tmp = (unsigned int *)malloc(size);

  BYTE *bits = (BYTE *)FreeImage_GetBits(bitmap);
  for (int y = 0; y < height; y++)
  {
    BYTE *pixel = (BYTE *)bits;
    for (int x = 0; x < width; x++)
    {
      int idx = ((y * width) + x) * 3;
      img[idx + 0] = pixel[FI_RGBA_RED];
      img[idx + 1] = pixel[FI_RGBA_GREEN];
      img[idx + 2] = pixel[FI_RGBA_BLUE];
      pixel += 3;
    }
    // next line
    bits += pitch;
  }

  memcpy(d_img, img, size);
  memcpy(d_tmp, img, size);

  unsigned *d_a, *d_b, *d_c;

  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  cudaMemcpy(d_a, d_img, size, cudaMemcpyHostToDevice);

  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid((height / 32) + 1, (width / 32) + 1, 1);

  gris<<<dimGrid, dimBlock>>>(d_a, width, height);

  cudaError_t cudaerr = cudaDeviceSynchronize();

  cudaMemcpy(d_img, d_a, size, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n",
           cudaGetErrorString(cudaerr));

  cudaMemcpy(d_b, d_img, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, d_tmp, size, cudaMemcpyHostToDevice);

  sobel<<<dimGrid, dimBlock>>>(d_b, d_c, width, height);

  cudaMemcpy(d_img, d_b, size, cudaMemcpyDeviceToHost);

  cudaerr = cudaDeviceSynchronize();
  if (cudaerr != cudaSuccess)
    printf("kernel launch failed with error \"%s\".\n",
           cudaGetErrorString(cudaerr));

  FIBITMAP *split = FreeImage_Rescale(bitmap, width / 2, height / 2, FILTER_BOX);

  // Copy back
  memcpy(img, d_img, size);

  bits = (BYTE *)FreeImage_GetBits(bitmap);
  for (int y = 0; y < height; y++)
  {
    BYTE *pixel = (BYTE *)bits;
    for (int x = 0; x < width; x++)
    {
      RGBQUAD newcolor;

      int idx = ((y * width) + x) * 3;
      newcolor.rgbRed = img[idx + 0];
      newcolor.rgbGreen = img[idx + 1];
      newcolor.rgbBlue = img[idx + 2];

      if (!FreeImage_SetPixelColor(bitmap, x, y, &newcolor))
      {
        fprintf(stderr, "(%d, %d) Fail...\n", x, y);
      }

      pixel += 3;
    }
    // next line
    bits += pitch;
  }

  if (FreeImage_Save(FIF_PNG, bitmap, PathDest, 0))
    cout << "Image successfully saved ! " << endl;
  FreeImage_DeInitialise(); // Cleanup !

  free(img);
  free(d_img);
  free(d_tmp);
}
