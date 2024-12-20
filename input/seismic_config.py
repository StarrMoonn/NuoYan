class SeismicConfig:
    """地震波模拟参数配置类"""
    
    def __init__(self):
        # ====== 网格参数 ======
        self.grid = {
            'nx': 401,          # 水平方向网格点数
            'nz': 401,          # 垂直方向网格点数
            'dx': 10,           # 水平方向网格间距(m)
            'dz': 10            # 垂直方向网格间距(m)
        }
        
        # ====== 时间步长参数 ======
        self.time = {
            'nt': 4001,         # 总时间步数
            'dt': 0.0005        # 时间步长(s)
        }
        
        # ====== PML边界参数 ======
        self.pml = {
            'use_pml_xmin': True,   # 是否使用左边界PML层
            'use_pml_xmax': True,   # 是否使用右边界PML层
            'use_pml_zmin': True,   # 是否使用上边界PML层
            'use_pml_zmax': True,   # 是否使用下边界PML层
            'npoints_pml': 10       # PML层的厚度(网格点数)
        }
        
        # ====== 震源参数 ======
        self.source = {
            'f0': 200.0e3,      # 震源主频(Hz)
            'factor': 1.0e7,     # 震源强度因子
            'position': {
                'x': None,       # 震源x位置(会被自动设置为nx//2)
                'z': None        # 震源z位置(会被自动设置为nz//2)
            },
            'angle': 0.0         # 震源力的方向角(度)
        }
        
        # ====== 检波器参数 ======
        self.receivers = {
            'n_receivers': 50,    # 检波器数量
            'first_position': {
                'x': 100,         # 第一个检波器x位置(网格点)
                'z': 50          # 第一个检波器z位置(网格点)
            },
            'spacing': {
                'dx': 4,          # 检波器x方向间距(网格点)
                'dz': 0           # 检波器z方向间距(网格点)
            }
        }
        
        # ====== 模型参数 ======
        self.model = {
            'scale_aniso': 1.0e10,   # 各向异性系数缩放因子(Pa)
            'density': 4000.0,        # 密度(kg/m³)
            # 刚度系数(Pa)
            'c11': 4.0,              # 会被自动乘以scale_aniso
            'c13': 3.8,              # 会被自动乘以scale_aniso
            'c33': 20.0,             # 会被自动乘以scale_aniso
            'c44': 2.0               # 会被自动乘以scale_aniso
        }
        
        # ====== 输出参数 ======
        self.output = {
            'display_interval': 100,  # 显示间隔(时间步数)
            'output_dir': "output"    # 输出目录
        }
        
        self._validate_and_update()
    
    def _validate_and_update(self):
        """验证并更新参数的内部一致性"""
        # 设置默认震源位置（如果未指定）
        if self.source['position']['x'] is None:
            self.source['position']['x'] = self.grid['nx'] // 2
        if self.source['position']['z'] is None:
            self.source['position']['z'] = self.grid['nz'] // 2
        
        # 验证网格参数
        if self.grid['nx'] <= 0 or self.grid['nz'] <= 0:
            raise ValueError("网格点数必须为正数")
        if self.grid['dx'] <= 0 or self.grid['dz'] <= 0:
            raise ValueError("网格间距必须为正数")
            
        # 验证时间步参数
        if self.time['nt'] <= 0 or self.time['dt'] <= 0:
            raise ValueError("时间步参数必须为正数")
            
        # 验证检波器参数
        if (self.receivers['first_position']['x'] >= self.grid['nx'] or
            self.receivers['first_position']['z'] >= self.grid['nz']):
            raise ValueError("检波器位置超出网格范围")

    def update(self, **kwargs):
        """更新配置参数
        
        使用示例:
        config.update(
            grid={'nx': 601, 'dx': 5},
            source={'f0': 100.0e3}
        )
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                current_dict = getattr(self, key)
                if isinstance(value, dict):
                    current_dict.update(value)
                else:
                    setattr(self, key, value)
            else:
                raise ValueError(f"未知的参数组: {key}")
        
        self._validate_and_update() 