mex文件说明

1.cuda版本mex
mex compute_wave_propagation_gpu.cu
compute_wave_propagation_gpu.mexw64 //windows系统


2.常规mex
mex compute_wave_propagation.cpp
compute_wave_propagation.mexmaca64  //macos系统
compute_wave_propagation.mexmw64    //windows系统

3.openmp并行版本的mex
mex compute_wave_propagation_omp.cpp COMPFLAGS="/openmp $COMPFLAGS"
compute_wave_propagation_omp.mexw64  //windows系统

4.GCD并行版本mex
mex LDFLAGS='\$LDFLAGS -framework Foundation -framework CoreFoundation' compute_wave_propagation_gcd.cpp
compute_wave_propagation_gcd.mexmaca64  //macos系统

