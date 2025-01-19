# MEX 文件编译说明

本文档说明了不同平台和并行方案下的 MEX 文件编译方法。

## 1. CUDA GPU 版本
适用于 NVIDIA GPU 加速计算：

mex compute_wave_propagation_gpu.cu

- 输出文件：
  - Windows: `compute_wave_propagation_gpu.mexw64`



## 2. CPU 基础版本
标准 C++ 实现：

mex compute_wave_propagation.cpp

- 输出文件：
  - macOS: `compute_wave_propagation.mexmaca64`
  - Windows: `compute_wave_propagation.mexw64`



## 3. OpenMP 并行版本
使用 OpenMP 进行 CPU 多线程并行：

mex compute_wave_propagation_omp.cpp COMPFLAGS="/openmp $COMPFLAGS"

- 输出文件：
  - Windows: `compute_wave_propagation_omp.mexw64`



## 4. GCD 并行版本
使用 Apple 的 Grand Central Dispatch 进行并行计算：

mex -O -largeArrayDims LDFLAGS='\$LDFLAGS -framework Foundation -framework CoreFoundation' compute_wave_propagation_gcd.cpp

- 输出文件：
  - macOS: `compute_wave_propagation_gcd.mexmaca64`

## 注意事项
- CUDA 版本需要安装 NVIDIA CUDA Toolkit
- OpenMP 版本在 Windows 下使用 MSVC 编译器
- GCD 版本仅支持 macOS 系统
- 建议在编译时使用 `-v` 参数查看详细信息
