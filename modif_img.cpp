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

  memcpy(d_img, img, 3 * width * height * sizeof(unsigned int));
  memcpy(d_tmp, img, 3 * width * height * sizeof(unsigned int));

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
  // FILTRE GRIS
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
  // unsigned int *d_tmp = (unsigned int *)malloc(size);

  memcpy(d_tmp, d_img, 3 * width * height * sizeof(unsigned int));

  /*
      SOBEL
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
      }*/

  /*
    int newHeight = height / 2;
    int newWidth = width / 2;

    FIBITMAP *newImage = FreeImage_Allocate(newWidth, newHeight, 24);

    unsigned *img_new = (unsigned *)malloc(3 * newWidth * newHeight * sizeof(unsigned int));

    for (int y = 0; y < newHeight; y++)
    {
      for (int x = 0; x < newWidth; x++)
      {
        int idx = ((y * newWidth) + x) * 3;

        float u = (float)x / (float)newWidth * (float)width;
        float v = (float)y / (float)newHeight * (float)height;

        int x1 = (int)u;
        int y1 = (int)v;
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        if (x2 >= width)
        {
          x2 = width - 1;
        }
        if (y2 >= height)
        {
          y2 = height - 1;
        }

        float a = u - (float)x1;
        float b = v - (float)y1;

        int idv1 = (((y1 * width) + x1) * 3);
        int idv2 = (((y1 * width) + x2) * 3);
        int idv3 = (((y2 * width) + x1) * 3);
        int idv4 = (((y2 * width) + x2) * 3);

        img_new[idx] = (BYTE)(img[idv1] * (1.0f - a) * (1.0f - b) + img[idv2] * a * (1.0f - b) + img[idv3] * (1.0f - a) * b + img[idv4] * a * b);
        img_new[idx + 1] = (BYTE)(img[idv1 + 1] * (1.0f - a) * (1.0f - b) + img[idv2 + 1] * a * (1.0f - b) + img[idv3 + 1] * (1.0f - a) * b + img[idv4 + 1] * a * b);
        img_new[idx + 2] = (BYTE)(img[idv1 + 2] * (1.0f - a) * (1.0f - b) + img[idv2 + 2] * a * (1.0f - b) + img[idv3 + 2] * (1.0f - a) * b + img[idv4 + 2] * a * b);

        // Définir la valeur du nouveau pixel dans la nouvelle image
      }
    }

    bits = (BYTE *)FreeImage_GetBits(newImage);
    for (int y = 0; y < newHeight; y++)
    {
      BYTE *pixel = (BYTE *)bits;
      for (int x = 0; x < newWidth; x++)
      {
        RGBQUAD newcolor;

        int idx = ((y * newWidth) + x) * 3;
        newcolor.rgbRed = img_new[idx + 0];
        newcolor.rgbGreen = img_new[idx + 1];
        newcolor.rgbBlue = img_new[idx + 2];

        if (!FreeImage_SetPixelColor(newImage, x, y, &newcolor))
        {
          fprintf(stderr, "(%d, %d) Fail...\n", x, y);
        }

        pixel += 3;
      }
      // next line
      bits += pitch;
    }*/

  /*if (FreeImage_Save(FIF_PNG, newImage, "resizetest.png", 0))
    cout << "Image successfully saved ! " << endl;
  FreeImage_DeInitialise(); // Cleanup !*/

  // Enregistrer la nouvelle image
  // FreeImage_Save(FIF_JPEG, newImage, "image_resized.jpg", 0);
  /*
    for (int i = 0; i < newHeight; i++)
    {
      for (int j = 0; j < newWidth; j++)
      {

        float originalX = (float)i / (float)newWidth * (float)width;
        float originalY = (float)j / (float)newHeight * (float)height;

        // Déterminer les coordonnées des pixels avoisinants nécessaires pour l'interpolation bilinéaire
        int x1 = (int)originalX;
        int y1 = (int)originalY;
        int x2 = x1 + 1;
        int y2 = y1 + 1;

        // Vérifier si les coordonnées sont dans les limites de l'image d'origine
        if (x2 >= width)
        {
          x2 = width - 1;
          x1 = x2 - 1;
        }
        if (y2 >= height)
        {
          y2 = height - 1;
          y1 = y2 - 1;
        }
        if (x1 < 0)
        {
          x1 = 0;
          x2 = 1;
        }
        if (y1 < 0)
        {
          y1 = 0;
          y2 = 1;
        }

        // Déterminer les coefficients d'interpolation bilinéaire
        float a = (float)x2 - originalX;
        float b = originalX - (float)x1;
        float c = (float)y2 - originalY;
        float d = originalY - (float)y1;

        // Déterminer les valeurs de chaque pixel avoisinant nécessaire pour l'interpolation bilinéaire
        RGBQUAD pixel1, pixel2, pixel3, pixel4;
        FreeImage_GetPixelColor(bitmap, x1, y1, &pixel1);
        FreeImage_GetPixelColor(bitmap, x2, y1, &pixel2);
        FreeImage_GetPixelColor(bitmap, x1, y2, &pixel3);
        FreeImage_GetPixelColor(bitmap, x2, y2, &pixel4);

        // Déterminer la valeur du nouveau pixel à partir de l'interpolation bilinéaire
        RGBQUAD newPixel;
        newPixel.rgbRed = (BYTE)(pixel1.rgbRed * a * c + pixel2.rgbRed * b * c + pixel3.rgbRed * a * d + pixel4.rgbRed * b * d);
        newPixel.rgbGreen = (BYTE)(pixel1.rgbGreen * a * c + pixel2.rgbGreen * b * c + pixel3.rgbGreen * a * d + pixel4.rgbGreen * b * d);
        newPixel.rgbBlue = (BYTE)(pixel1.rgbBlue * a * c + pixel2.rgbBlue * b * c + pixel3.rgbBlue * a * d + pixel4.rgbBlue * b * d);
        FreeImage_SetPixelColor(newImage, i, j, &newPixel);

        /*

        int x = i * 2;
        int y = j * 2;

        int x_min = ceil(x);
        int x_max = floor(x);

        int y_min = ceil(y);
        int y_max = floor(y);

        int idx = ((j * width) + i) * 3;

        int idv1 = ((y_min * width) + x_min) * 3;
        int idv2 = ((y_min * width) + x_max) * 3;
        int idv3 = ((y_max * width) + x_max) * 3;
        int idv4 = ((y_max * width) + x_max) * 3;

        int q1 = img_tmp[idv1] * (x_max - x) + img_tmp[idv2] * (x - x_min);

        int q2 = img_tmp[idv3] * (x_max - x) + img_tmp[idv4] * (x - x_min);

        int q = q1 * (y_max - y) + q2 * (y - y_min);

        img[idx] = img_tmp[q];

        q1 = img_tmp[idv1 + 1] * (x_max - x) + img_tmp[idv2 + 1] * (x - x_min);

        q2 = img_tmp[idv3 + 1] * (x_max - x) + img_tmp[idv4 + 1] * (x - x_min);

        q = q1 * (y_max - y) + q2 * (y - y_min);
        img[idx + 1] = img_tmp[q];
        img[idx + 2] = img_tmp[q + 2];*/
  // }
  // }
  // FreeImage_Save(FIF_PNG, newImage, "image_resized.png", 0);

  /*
    for (int y = 0; y < height; y++)
    {
      for (int x = 0; x < width; x++)
      {

        int idx = ((y * width) + x) * 3;

        d_img[idx + 0] = 255 - d_img[idx];
        d_img[idx + 1] = 255 - d_img[idx + 1];
        d_img[idx + 2] = 255 - d_img[idx + 2];
      }
    }*/

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

      int gx = -d_tmp[idv1] + d_tmp[idv2];

      int gy = -d_tmp[idv3] + d_tmp[idv4];

      int gn = sqrt(gx * gx + gy * gy);

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
  // free(d_tmp);
}
