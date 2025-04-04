# 核心代码说明

## 代码来源与实现
- 基于 `seismic_CPML_2D_anisotropic.f90` 程序移植的正演模块
- 中海油MATLAB参考代码已复制到 `external` 文件夹备用

## 技术特点
1. **差分方案**
   - 采用索引差异实现的二阶交错网格
   - 不涉及差分系数计算

2. **边界条件**
   - 使用CPML（Convolutional Perfectly Matched Layer）完美匹配层
   - 无需考虑其他边界条件适配

3. **伴随源处理**
   - 当前：直接使用残差作为伴随源
   - 中海油方案：使用包络差（见`cross_resid1.m`和`cross_resid2.m`）
   - 优势：基于频谱而非波形相位，更适合实际数据处理

4. **目标函数**
   - 当前：波形残差二范数
   - 中海油方案：互相关残差（`objfun1.m`和`objfun12.m`）
     - 基于波形的标准版
     - 基于包络差的加强版

5. **优化算法**
   - 梯度下降法：`GradientDescentOptimizer.m`
   - L-BFGS法：`LBFGSOptimizer.m`
   - Fletcher-Reeves共轭梯度法：`FletcherReevesCGOptimizer.m`
     - 基于中海油参考代码移植
     - 结合最速下降法和共轭梯度法
     - 适用于大规模反演问题

## 待开发功能
1. **实际数据处理**
   - 集成实际SEG-Y数据读取功能
   - 可考虑使用开源的SegyMAT工具包

2. **子波提取**
   - 从实际数据反演提取地震子波
   - 需要基于震源位置的时窗处理
   - 实现数据加窗过滤

## 时窗处理说明
### 地震记录基本结构
```
时间 ↓     偏移距 →
0   ┌───────────────┐
    │       ▲       │  ▲ = 震源位置
    │      / \      │
    │     /   \     │  /\= 实际地震信号
    │    /     \    │     （A字型）
    │   /       \   │
nt  └───────────────┘
    道1          道N
```

### 窗函数类型
- 斜率窗（当前使用）
- 矩形窗
- 高斯窗
- 汉宁窗

### 斜率窗示例
```
时间 ↓     偏移距 →
0   ┌───────────────┐
    │///    ▲    \\\│  ▲ = 震源位置
    │///   / \   \\\│
    │///  /   \  \\\│  /// = 置零区域
    │/// /     \ \\\│
    │///         \\\│
nt  └───────────────┘
```

### 斜率控制
- `i = k2*abs(sx-j)`
  - k2：斜率参数（通常取6或7）
  - sx：震源道号
  - j：当前道号

### 时窗处理目的
1. 突出直达波信息
2. 减少多次波干扰
3. 提高反演质量

## 后续优化方向
1. 基于包络的伴随源计算
2. 包络归一化的互相关计算
3. 改进敛散性判断标准
4. 移植基于最速下降法结合共轭梯度法
