default: all

all:
	g++ -I${HOME}/softs/FreeImage/include modif_img.cpp -L${HOME}/softs/FreeImage/lib/ -lfreeimage -g -o modif_img -lm
# nvcc -I${HOME}/softs/FreeImage/include modif_img.cu -L${HOME}/softs/FreeImage/lib/ -lfreeimage -o modif_cuda

clean:
	rm -f *.o modif_img.exe
