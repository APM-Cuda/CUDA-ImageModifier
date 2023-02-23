#include <iostream>
#include <string.h>
#include "FreeImage.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 1920
#define HEIGHT 1024
#define BPP 24 // Since we're outputting three 8 bit RGB values

using namespace std;

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

  memcpy(d_img, img, 3 * width * height * sizeof(unsigned int));

  // Kernel
  /*
    for (int y = 0; y < height; y++)
    {
      for (int x = 0; x < width; x++)
      {
        int ida = ((y * width) + x) * 3;
        int idb = ((width * height) - ((y * width) + x)) * 3;
        d_img[ida + 0] = d_tmp[idb + 0];
        d_img[ida + 1] = d_tmp[idb + 1];
        d_img[ida + 2] = d_tmp[idb + 2];
      }
    }*/

  /*
    for (int y = 0; y < height; y++)
    {
      for (int x = 0; x < width; x++)
      {

        int idx = ((y * width) + x) * 3;
        d_img[idx + 0] /= 2;
        d_img[idx + 1] /= 4;
        d_img[idx + 2] = 0xFF / 1.5;
      }
    }*/

  /*

  for (int y = height / 2; y < height; y++)
  {
    for (int x = width / 2; x < width; x++)
    {
      if (x >= ((width / 2) + (width / 2 * 0.25)) || y >= ((height / 2) + (height / 2 * 0.35)))
      {
        int idx = ((y * width) + x) * 3;
        d_img[idx + 0] = 0xFF - d_img[idx + 0];
        d_img[idx + 1] = 0xFF / 2;
        d_img[idx + 2] /= 4;
      }
    }
  }

  for (int y = height / 2; y < height; y++)
  {
    for (int x = 0; x < width / 2; x++)
    {
      if (x < (width / 2 * 0.75) || y >= (height / 2) + (height / 2 * 0.35))
      {
        int idx = ((y * width) + x) * 3;
        d_img[idx + 0] = 0xFF / 2;
        d_img[idx + 1] /= 2;
        d_img[idx + 2] /= 2;
      }
    }
  }

  for (int y = 0; y < height / 2; y++)
  {
    for (int x = width / 2; x < width; x++)
    {
      if (x >= ((width / 2) + (width / 2 * 0.25)) || y < (height / 2 * 0.65))
      {
        int idx = ((y * width) + x) * 3;
        int grey = d_img[idx + 0] * 0.299 + d_img[idx + 1] * 0.587 + d_img[idx + 2] * 0.114;
        d_img[idx + 0] = 0xFF - d_img[idx + 0];
        d_img[idx + 1] = 0xFF - d_img[idx + 1];
        d_img[idx + 2] = 0xFF - d_img[idx + 2];
        d_img[idx + 0] = grey;
        d_img[idx + 1] = grey;
        d_img[idx + 2] = grey;
      }
    }
  }*/

  // for (int k = 0; k < 100; k++)

  // Flou
  /*
      for (int y = 0; y < height; y++)
      {
        for (int x = 0; x < width; x++)
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
  */
  /* FLOU GRIS
    for (int y = 0; y < height; y++)
    {
      for (int x = 0; x < width; x++)
      {

        float count = 0, count1 = 0, count2 = 0, c = 0;
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

        count *= 0.299 + count1 * 0.587 + count2 * 0.114;

        c++;

        d_img[idx + 0] = count;
        d_img[idx + 1] = count;
        d_img[idx + 2] = count;
      }
    }*/

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {

      int idx = ((y * width) + x) * 3;
      int grey = d_img[idx + 0] * 0.299 + d_img[idx + 1] * 0.587 + d_img[idx + 2] * 0.114;

      d_img[idx + 0] = grey;
      d_img[idx + 1] = grey;
      d_img[idx + 2] = grey;
    }
  }
  unsigned int *d_tmp = (unsigned int *)malloc(size);

  memcpy(d_tmp, d_img, 3 * width * height * sizeof(unsigned int));

  for (int y = 0; y < height; y++)
  {
    for (int x = 0; x < width; x++)
    {
      int idx = ((y * width) + x) * 3;

      if (y == 0 || x == 0 || y == height - 1 || x == width - 1)
      {
        d_img[idx] = 0;
        d_img[idx + 1] = 0;
        d_img[idx + 2] = 0;
        continue;
      }

      int idv1 = (y * width + (x - 1) * 3);
      int idv2 = (y * width + (x + 1) * 3);
      int idv3 = (((y - 1) * width + x) * 3);
      int idv4 = (((y + 1) * width + x) * 3);
      int idv5 = (((y - 1) * width) + (x - 1)) * 3;
      int idv6 = (((y - 1) * width) + (x + 1)) * 3;
      int idv7 = (((y + 1) * width) + (x - 1)) * 3;
      int idv8 = (((y + 1) * width) + (x + 1)) * 3;

      int gx = -d_tmp[idv7] - d_tmp[idv5] - 2 * d_tmp[idv1] + d_tmp[idv8] + d_tmp[idv6] + 2 * d_tmp[idv2];

      int gy = -d_tmp[idv7] - d_tmp[idv8] - 2 * d_tmp[idv4] + d_tmp[idv5] + d_tmp[idv6] + 2 * d_tmp[idv3];

      int gn = (int)sqrt(gx * gx + gy * gy);

      d_img[idx + 0] = gn;

      d_img[idx + 1] = gn;

      d_img[idx + 2] = gn;
    }
  }

  // Copy back
  memcpy(img, d_img, 3 * width * height * sizeof(unsigned int));

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
