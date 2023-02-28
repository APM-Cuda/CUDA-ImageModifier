# Cuda Modifier
This project is based on the use of CUDA and the FreeImage library.
The objectives of this project are multiple and include the black-and-white effect, edge detection, colour saturation, among others. To do this, we modify an image by performing several transformations on all or part of it. An image is an array of pixels encoded according to the RGB standard, where each pixel has three components: R (red), G (green) and B (blue).
FreeImage is an open-source library that provides functionality for image manipulation, including loading, saving and converting a wide variety of image formats.

## How to ?

This program uses the arguments given at program startup to apply filters and effects to the image, if a filter option is not given (if the filter needs one) the default value will be used.

./modif_cuda --<filter> <option>

List of availables options:
- sobel <level> | Apply Sobel filter for edge detection
     Level: Integer number to define black to white in the result
     Default: 5000

- canny <level> | Apply Canny filter for edge detection
     Level: Integer number to define black to white in the result. 
     Default: 1

- popart | Apply a pop-art filter with 4 same images with 4 differents saturation filter (blue/red/green/grey)
  
- resize <factor> | Resize image
     factor: Double number for the resizing factor, it will be (width/height*factor), to divide the size, use > 1 numbers, ie: size/2 = 0.5. Default : 0.5

- only <color> | Keep only one colour on the pixels
      Colors: RED/BLUE/GREEN/GREENBLUE/GREENRED/REDBLUE

- saturation <color> | Saturate a colour on the pixels
      Colors: RED/BLUE/GREEN/CYAN/GREY/YELLOW/MAGENTA
    
- sym | Apply an horizontal symmetry
  
- blur <iterations> | Apply a blur effect
      Iterations: Integer number, how many time the blur effect will be done, at minimum 20 iterations is recommended
      Default: 20
    
- output <name> | Custom output image name
      Name: String value, will still be stored in the ./output/ directory
      Default: new_img.png

- input <name> | Custom input image name
      Name: String value, will still be retrieved from the ./input/ directory
      Default: img.png

- help | Prints help message and exits program

/!\ The order of the arguments is necessary, the actions will be executed in order. If for example, you perform a filter before modifying the input file, then the filter will be done on the base image.

Example: ./modif_cuda --saturation RED

## Existing filters

-  Sobel 
-  Canny 
-  Resize 
-  Only
    -   Red 
    -   Blue 
    -   Green 
    -   Green-Red
    -   Green-Blue
    -   Red-Blue
- Diapositive
- Blur 
- Saturation
  - Blue 
  - Red 
  - Green 
  - Cyan 
  - Grey
  - Yellow
  - Magenta
- Horizontal symmetry 
- Pop-art 