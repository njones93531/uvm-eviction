all: touch_pages

touch_pages: touch_pages.cu Makefile
	nvcc -arch=sm_70 -Xptxas -v -Xcompiler "-Wall -Wextra -O2 -g" -O0 -G -Xptxas -dlcm=cg touch_pages.cu -o touch_pages
	cuobjdump -ptx -sass touch_pages > touch_pages.dump

clean: rm -f touch_pages
