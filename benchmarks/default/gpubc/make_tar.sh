mkdir gpu_BC; 
cp Makefile *.cpp *.hpp *.c *.h *.cu LICENSE README.txt ChangeLog gpu_BC ;
tar -zcvf gpu_BC.tar.gz gpu_BC;
rm gpu_BC -r
