import numpy as np
import matplotlib.pyplot as plt
import os
import cupy as cp

class SeismicCPML2DAniso:
    def __init__(self):
        # Grid parameters
        self.NX = 401
        self.NY = 401
        
        # Grid spacing
        self.DELTAX = 0.0625e-2
        self.DELTAY = self.DELTAX
        
        # PML flags and parameters  
        self.USE_PML_XMIN = True
        self.USE_PML_XMAX = True
        self.USE_PML_YMIN = True
        self.USE_PML_YMAX = True
        self.NPOINTS_PML = 10
        
        # Material properties (Model I from Becache, Fauqueux and Joly)
        self.scale_aniso = 1.0e10
        self.c11 = 4.0 * self.scale_aniso
        self.c12 = 3.8 * self.scale_aniso  
        self.c22 = 20.0 * self.scale_aniso
        self.c33 = 2.0 * self.scale_aniso
        self.rho = 4000.0
        self.f0 = 200.0e3
        
        # Time stepping parameters
        self.NSTEP = 3000  # Changed to match Fortran version
        self.DELTAT = 50.0e-9
        
        # Source parameters
        self.t0 = 1.20/self.f0
        self.factor = 1.0e7
        self.ISOURCE = self.NX // 2
        self.JSOURCE = self.NY // 2
        self.xsource = (self.ISOURCE - 1) * self.DELTAX
        self.ysource = (self.JSOURCE - 1) * self.DELTAY
        self.ANGLE_FORCE = 0.0
        
        # Receiver parameters
        self.NREC = 50  # Number of receivers
        self.first_rec_x = 100  # First receiver x position (grid points)
        self.first_rec_z = 50   # First receiver z position (grid points)
        self.rec_dx = 4         # Receiver x spacing (grid points)
        self.rec_dz = 0         # Receiver z spacing (grid points)
        
        # Initialize receiver arrays
        self.rec_x = np.zeros(self.NREC, dtype=np.int32)
        self.rec_z = np.zeros(self.NREC, dtype=np.int32)
        
        # Initialize seismogram arrays
        self.seismogram_vx = np.zeros((self.NSTEP, self.NREC))
        self.seismogram_vz = np.zeros((self.NSTEP, self.NREC))
        
        # Display parameters
        self.IT_DISPLAY = 100
        
        # Constants
        self.PI = cp.pi
        self.DEGREES_TO_RADIANS = self.PI / 180.0
        self.ZERO = cp.float64(0.0)
        self.HUGEVAL = cp.float64(1.0e+30)
        self.STABILITY_THRESHOLD = cp.float64(1.0e+25)
        
        # PML parameters
        self.NPOWER = cp.float64(2.0)
        self.K_MAX_PML = cp.float64(1.0)
        self.ALPHA_MAX_PML = cp.float64(2.0 * self.PI * (self.f0/2.0))
        
        # Initialize arrays
        self.initialize_arrays()
        
        # Setup receivers
        self.setup_receivers() 

    def initialize_arrays(self):
        """Initialize all the arrays needed for the simulation on GPU"""
        # Main field arrays
        self.vx = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.vy = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.sigmaxx = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.sigmayy = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.sigmaxy = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        
        # Memory variables for PML
        self.memory_dvx_dx = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.memory_dvx_dy = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.memory_dvy_dx = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.memory_dvy_dy = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.memory_dsigmaxx_dx = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.memory_dsigmayy_dy = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.memory_dsigmaxy_dx = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        self.memory_dsigmaxy_dy = cp.zeros((self.NX, self.NY), dtype=cp.float64)
        
        # 1D arrays for damping profiles
        self.d_x = cp.zeros(self.NX, dtype=cp.float64)
        self.d_x_half = cp.zeros(self.NX, dtype=cp.float64)
        self.K_x = cp.ones(self.NX, dtype=cp.float64)
        self.K_x_half = cp.ones(self.NX, dtype=cp.float64)
        self.alpha_x = cp.zeros(self.NX, dtype=cp.float64)
        self.alpha_x_half = cp.zeros(self.NX, dtype=cp.float64)
        self.a_x = cp.zeros(self.NX, dtype=cp.float64)
        self.a_x_half = cp.zeros(self.NX, dtype=cp.float64)
        self.b_x = cp.zeros(self.NX, dtype=cp.float64)
        self.b_x_half = cp.zeros(self.NX, dtype=cp.float64)
        
        self.d_y = cp.zeros(self.NY, dtype=cp.float64)
        self.d_y_half = cp.zeros(self.NY, dtype=cp.float64)
        self.K_y = cp.ones(self.NY, dtype=cp.float64)
        self.K_y_half = cp.ones(self.NY, dtype=cp.float64)
        self.alpha_y = cp.zeros(self.NY, dtype=cp.float64)
        self.alpha_y_half = cp.zeros(self.NY, dtype=cp.float64)
        self.a_y = cp.zeros(self.NY, dtype=cp.float64)
        self.a_y_half = cp.zeros(self.NY, dtype=cp.float64)
        self.b_y = cp.zeros(self.NY, dtype=cp.float64)
        self.b_y_half = cp.zeros(self.NY, dtype=cp.float64)

    def setup_receivers(self):
        """Setup receiver positions"""
        for i in range(self.NREC):
            self.rec_x[i] = self.first_rec_x + i * self.rec_dx
            self.rec_z[i] = self.first_rec_z + i * self.rec_dz
            
        # Check if receivers are within grid bounds
        if np.any(self.rec_x >= self.NX) or np.any(self.rec_z >= self.NY):
            raise ValueError("Receiver positions exceed grid dimensions")

    def setup_pml_boundary(self):
        """Setup the PML boundary conditions"""
        
        # Check stability of PML model for anisotropic material
        aniso_stability_criterion = ((self.c12 + self.c33)**2 - self.c11*(self.c22-self.c33)) * \
                                   ((self.c12 + self.c33)**2 + self.c33*(self.c22-self.c33))
        print(f'PML anisotropy stability criterion from Becache et al. 2003 = {aniso_stability_criterion}')
        if aniso_stability_criterion > 0.0 and (self.USE_PML_XMIN or self.USE_PML_XMAX or 
                                              self.USE_PML_YMIN or self.USE_PML_YMAX):
            print('WARNING: PML model mathematically intrinsically unstable for this anisotropic material for condition 1')
        
        aniso2 = (self.c12 + 2*self.c33)**2 - self.c11*self.c22
        print(f'PML aniso2 stability criterion from Becache et al. 2003 = {aniso2}')
        if aniso2 > 0.0 and (self.USE_PML_XMIN or self.USE_PML_XMAX or 
                             self.USE_PML_YMIN or self.USE_PML_YMAX):
            print('WARNING: PML model mathematically intrinsically unstable for this anisotropic material for condition 2')
        
        aniso3 = (self.c12 + self.c33)**2 - self.c11*self.c22 - self.c33**2
        print(f'PML aniso3 stability criterion from Becache et al. 2003 = {aniso3}')
        if aniso3 > 0.0 and (self.USE_PML_XMIN or self.USE_PML_XMAX or 
                             self.USE_PML_YMIN or self.USE_PML_YMAX):
            print('WARNING: PML model mathematically intrinsically unstable for this anisotropic material for condition 3')

    def setup_pml_boundary_y(self):
        """Setup the PML boundary conditions for y direction"""
        # Calculate quasi_cp_max for d0 computation
        quasi_cp_max = cp.maximum(cp.sqrt(self.c22/self.rho), cp.sqrt(self.c11/self.rho))
        
        # Define profile of absorption in PML region
        thickness_PML_y = self.NPOINTS_PML * self.DELTAY
        
        # Reflection coefficient
        Rcoef = cp.float64(0.001)
        
        # Compute d0
        d0_y = -(self.NPOWER + 1) * quasi_cp_max * cp.log(Rcoef) / (2.0 * thickness_PML_y)
        
        # Check NPOWER
        if self.NPOWER < 1:
            raise ValueError('NPOWER must be greater than 1')
        
        # Compute d0
        d0_y = -(self.NPOWER + 1) * quasi_cp_max * cp.log(Rcoef) / (2.0 * thickness_PML_y)
        
        print(f'd0_y = {d0_y}')
        
        # Setup damping profiles
        yoriginleft = thickness_PML_y
        yoriginright = (self.NY-1)*self.DELTAY - thickness_PML_y
        
        # Create arrays for y values
        y_vals = cp.arange(self.NY, dtype=cp.float64) * self.DELTAY
        y_vals_half = y_vals + self.DELTAY/2.0
        
        # Left edge
        if self.USE_PML_YMIN:
            abscissa_in_PML = yoriginleft - y_vals
            mask = abscissa_in_PML >= 0.0
            abscissa_normalized = cp.where(mask, abscissa_in_PML / thickness_PML_y, 0.0)
            self.d_y = cp.where(mask, d0_y * abscissa_normalized**self.NPOWER, self.d_y)
            self.K_y = cp.where(mask, 1.0 + (self.K_MAX_PML - 1.0) * abscissa_normalized**self.NPOWER, self.K_y)
            self.alpha_y = cp.where(mask, self.ALPHA_MAX_PML * (1.0 - abscissa_normalized), self.alpha_y)
            
            abscissa_in_PML_half = yoriginleft - y_vals_half
            mask_half = abscissa_in_PML_half >= 0.0
            abscissa_normalized_half = cp.where(mask_half, abscissa_in_PML_half / thickness_PML_y, 0.0)
            self.d_y_half = cp.where(mask_half, d0_y * abscissa_normalized_half**self.NPOWER, self.d_y_half)
            self.K_y_half = cp.where(mask_half, 1.0 + (self.K_MAX_PML - 1.0) * abscissa_normalized_half**self.NPOWER, self.K_y_half)
            self.alpha_y_half = cp.where(mask_half, self.ALPHA_MAX_PML * (1.0 - abscissa_normalized_half), self.alpha_y_half)
        
        # Right edge
        if self.USE_PML_YMAX:
            abscissa_in_PML = y_vals - yoriginright
            mask = abscissa_in_PML >= 0.0
            abscissa_normalized = cp.where(mask, abscissa_in_PML / thickness_PML_y, 0.0)
            self.d_y = cp.where(mask, d0_y * abscissa_normalized**self.NPOWER, self.d_y)
            self.K_y = cp.where(mask, 1.0 + (self.K_MAX_PML - 1.0) * abscissa_normalized**self.NPOWER, self.K_y)
            self.alpha_y = cp.where(mask, self.ALPHA_MAX_PML * (1.0 - abscissa_normalized), self.alpha_y)
            
            abscissa_in_PML_half = y_vals_half - yoriginright
            mask_half = abscissa_in_PML_half >= 0.0
            abscissa_normalized_half = cp.where(mask_half, abscissa_in_PML_half / thickness_PML_y, 0.0)
            self.d_y_half = cp.where(mask_half, d0_y * abscissa_normalized_half**self.NPOWER, self.d_y_half)
            self.K_y_half = cp.where(mask_half, 1.0 + (self.K_MAX_PML - 1.0) * abscissa_normalized_half**self.NPOWER, self.K_y_half)
            self.alpha_y_half = cp.where(mask_half, self.ALPHA_MAX_PML * (1.0 - abscissa_normalized_half), self.alpha_y_half)
        
        # Calculate b and a coefficients
        self.b_y = cp.exp(-(self.d_y / self.K_y + self.alpha_y) * self.DELTAT)
        self.b_y_half = cp.exp(-(self.d_y_half / self.K_y_half + self.alpha_y_half) * self.DELTAT)
        
        mask = cp.abs(self.d_y) > 1.0e-6
        self.a_y = cp.where(mask,
                           self.d_y * (self.b_y - 1.0) / (self.K_y * (self.d_y + self.K_y * self.alpha_y)),
                           self.a_y)
        
        mask_half = cp.abs(self.d_y_half) > 1.0e-6
        self.a_y_half = cp.where(mask_half,
                                self.d_y_half * (self.b_y_half - 1.0) / (self.K_y_half * (self.d_y_half + self.K_y_half * self.alpha_y_half)),
                                self.a_y_half)

    def compute_stress(self):
        """Compute stress sigma and update memory variables"""
        # Compute stress for interior points using vectorized operations
        value_dvx_dx = (self.vx[1:,:] - self.vx[:-1,:]) / self.DELTAX
        value_dvy_dy = cp.zeros_like(value_dvx_dx)
        value_dvy_dy[:,1:] = (self.vy[:-1,1:] - self.vy[:-1,:-1]) / self.DELTAY
        
        self.memory_dvx_dx[:-1,:] = (self.b_x_half[:-1,None] * self.memory_dvx_dx[:-1,:] + 
                                    self.a_x_half[:-1,None] * value_dvx_dx)
        self.memory_dvy_dy[:-1,1:] = (self.b_y[1:,None].T * self.memory_dvy_dy[:-1,1:] + 
                                     self.a_y[1:,None].T * value_dvy_dy[:,1:])
        
        value_dvx_dx = value_dvx_dx / self.K_x_half[:-1,None] + self.memory_dvx_dx[:-1,:]
        value_dvy_dy[:,1:] = value_dvy_dy[:,1:] / self.K_y[1:,None].T + self.memory_dvy_dy[:-1,1:]
        
        self.sigmaxx[:-1,1:] += (self.c11 * value_dvx_dx[:,1:] + self.c12 * value_dvy_dy[:,1:]) * self.DELTAT
        self.sigmayy[:-1,1:] += (self.c12 * value_dvx_dx[:,1:] + self.c22 * value_dvy_dy[:,1:]) * self.DELTAT

        # Compute shear stress using vectorized operations
        value_dvy_dx = (self.vy[1:,:-1] - self.vy[:-1,:-1]) / self.DELTAX
        value_dvx_dy = (self.vx[1:,1:] - self.vx[1:,:-1]) / self.DELTAY
        
        self.memory_dvy_dx[1:,:-1] = (self.b_x[1:,None] * self.memory_dvy_dx[1:,:-1] + 
                                     self.a_x[1:,None] * value_dvy_dx)
        self.memory_dvx_dy[1:,:-1] = (self.b_y_half[:-1,None].T * self.memory_dvx_dy[1:,:-1] + 
                                     self.a_y_half[:-1,None].T * value_dvx_dy)
        
        value_dvy_dx = value_dvy_dx / self.K_x[1:,None] + self.memory_dvy_dx[1:,:-1]
        value_dvx_dy = value_dvx_dy / self.K_y_half[:-1,None].T + self.memory_dvx_dy[1:,:-1]
        
        self.sigmaxy[1:,:-1] += self.c33 * (value_dvy_dx + value_dvx_dy) * self.DELTAT

    def compute_velocity(self):
        """Compute velocity and update memory variables"""
        # Compute velocity components using vectorized operations
        value_dsigmaxx_dx = (self.sigmaxx[1:,1:] - self.sigmaxx[:-1,1:]) / self.DELTAX
        value_dsigmaxy_dy = (self.sigmaxy[1:,1:] - self.sigmaxy[1:,:-1]) / self.DELTAY
        
        self.memory_dsigmaxx_dx[1:,1:] = (self.b_x[1:,None] * self.memory_dsigmaxx_dx[1:,1:] + 
                                         self.a_x[1:,None] * value_dsigmaxx_dx)
        self.memory_dsigmaxy_dy[1:,1:] = (self.b_y[1:,None].T * self.memory_dsigmaxy_dy[1:,1:] + 
                                         self.a_y[1:,None].T * value_dsigmaxy_dy)
        
        value_dsigmaxx_dx = value_dsigmaxx_dx / self.K_x[1:,None] + self.memory_dsigmaxx_dx[1:,1:]
        value_dsigmaxy_dy = value_dsigmaxy_dy / self.K_y[1:,None].T + self.memory_dsigmaxy_dy[1:,1:]
        
        self.vx[1:,1:] += (value_dsigmaxx_dx + value_dsigmaxy_dy) * self.DELTAT / self.rho
        
        value_dsigmaxy_dx = (self.sigmaxy[1:,:-1] - self.sigmaxy[:-1,:-1]) / self.DELTAX
        value_dsigmayy_dy = (self.sigmayy[:-1,1:] - self.sigmayy[:-1,:-1]) / self.DELTAY
        
        self.memory_dsigmaxy_dx[:-1,:-1] = (self.b_x_half[:-1,None] * self.memory_dsigmaxy_dx[:-1,:-1] + 
                                           self.a_x_half[:-1,None] * value_dsigmaxy_dx)
        self.memory_dsigmayy_dy[:-1,:-1] = (self.b_y_half[:-1,None].T * self.memory_dsigmayy_dy[:-1,:-1] + 
                                           self.a_y_half[:-1,None].T * value_dsigmayy_dy)
        
        value_dsigmaxy_dx = value_dsigmaxy_dx / self.K_x_half[:-1,None] + self.memory_dsigmaxy_dx[:-1,:-1]
        value_dsigmayy_dy = value_dsigmayy_dy / self.K_y_half[:-1,None].T + self.memory_dsigmayy_dy[:-1,:-1]
        
        self.vy[:-1,:-1] += (value_dsigmaxy_dx + value_dsigmayy_dy) * self.DELTAT / self.rho

    def add_source(self, it):
        """Add the source (force vector located at a given grid point)"""
        a = self.PI * self.PI * self.f0 * self.f0
        t = (it-1) * self.DELTAT
        
        # First derivative of a Gaussian
        source_term = -self.factor * 2.0 * a * (t-self.t0) * cp.exp(-a*(t-self.t0)**2)
        
        force_x = cp.sin(self.ANGLE_FORCE * self.DEGREES_TO_RADIANS) * source_term
        force_y = cp.cos(self.ANGLE_FORCE * self.DEGREES_TO_RADIANS) * source_term
        
        # Define location of the source
        i = self.ISOURCE
        j = self.JSOURCE
        
        self.vx[i,j] += force_x * self.DELTAT / self.rho
        self.vy[i,j] += force_y * self.DELTAT / self.rho

    def record_seismograms(self, it):
        """Record seismogram data at receiver locations"""
        # Move data from GPU to CPU for recording
        vx_cpu = cp.asnumpy(self.vx)
        vy_cpu = cp.asnumpy(self.vy)
        
        # Record velocities at each receiver position
        for i in range(self.NREC):
            self.seismogram_vx[it-1, i] = vx_cpu[self.rec_x[i], self.rec_z[i]]
            self.seismogram_vz[it-1, i] = vy_cpu[self.rec_x[i], self.rec_z[i]]

    def apply_boundary_conditions(self):
        """Apply Dirichlet boundary conditions (rigid boundaries)"""
        self.vx[0,:] = self.ZERO
        self.vx[-1,:] = self.ZERO
        self.vx[:,0] = self.ZERO
        self.vx[:,-1] = self.ZERO
        
        self.vy[0,:] = self.ZERO
        self.vy[-1,:] = self.ZERO
        self.vy[:,0] = self.ZERO
        self.vy[:,-1] = self.ZERO

    def output_info(self, it):
        """Output information about the simulation status"""
        # Move data to CPU for visualization
        vx_cpu = cp.asnumpy(self.vx)
        vy_cpu = cp.asnumpy(self.vy)
        
        velocnorm = np.max(np.sqrt(vx_cpu**2 + vy_cpu**2))
        print(f'Time step # {it} out of {self.NSTEP}')
        print(f'Time: {(it-1)*self.DELTAT:.6f} seconds')
        print(f'Max norm velocity vector V (m/s) = {velocnorm}')
        print()
        
        if velocnorm > self.STABILITY_THRESHOLD:
            raise RuntimeError('code became unstable and blew up')
        
        self.create_color_image(vx_cpu, it, 1)
        self.create_color_image(vy_cpu, it, 2)

    def plot_seismograms(self):
        """Plot seismograms for all receivers"""
        time = np.arange(self.NSTEP) * self.DELTAT
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Plot horizontal component seismogram
        plt.subplot(211)
        plt.imshow(self.seismogram_vx, aspect='auto', cmap='seismic',
                  extent=[0, self.NREC-1, self.NSTEP*self.DELTAT*1000, 0])
        plt.colorbar(label='Amplitude')
        plt.title('Horizontal Component Seismogram')
        plt.xlabel('Receiver number')
        plt.ylabel('Time (ms)')
        
        # Plot vertical component seismogram
        plt.subplot(212)
        plt.imshow(self.seismogram_vz, aspect='auto', cmap='seismic',
                  extent=[0, self.NREC-1, self.NSTEP*self.DELTAT*1000, 0])
        plt.colorbar(label='Amplitude')
        plt.title('Vertical Component Seismogram')
        plt.xlabel('Receiver number')
        plt.ylabel('Time (ms)')
        
        plt.tight_layout()
        plt.savefig('seismograms.png')
        plt.close()

    def create_color_image(self, image_data_2D, it, field_number):
        """Create a color image of a given vector component"""
        # Parameters for visualization
        POWER_DISPLAY = 0.30  # non linear display to enhance small amplitudes
        cutvect = 0.01       # amplitude threshold above which we draw the color point
        WHITE_BACKGROUND = True
        width_cross = 5      # size of cross and square in pixels
        thickness_cross = 1
        
        # Create figure name based on field number
        if field_number == 1:
            field_name = 'Vx'
        else:
            field_name = 'Vy'
        
        fig_name = f'image{it:06d}_{field_name}.png'
        
        # Compute maximum amplitude
        max_amplitude = np.max(np.abs(image_data_2D))
        
        # Create RGB array for image
        img = np.zeros((self.NY, self.NX, 3))
        
        # Fill the image array
        for iy in range(self.NY-1, -1, -1):
            for ix in range(self.NX):
                # Normalize value to [-1,1]
                normalized_value = image_data_2D[ix,iy] / max_amplitude
                
                # Clip values to [-1,1]
                normalized_value = np.clip(normalized_value, -1.0, 1.0)
                
                # Draw source location
                if ((ix >= self.ISOURCE - width_cross and ix <= self.ISOURCE + width_cross and 
                     iy >= self.JSOURCE - thickness_cross and iy <= self.JSOURCE + thickness_cross) or
                    (ix >= self.ISOURCE - thickness_cross and ix <= self.ISOURCE + thickness_cross and
                     iy >= self.JSOURCE - width_cross and iy <= self.JSOURCE + width_cross)):
                    img[iy,ix] = [1.0, 0.616, 0.0]  # Orange
                
                # Draw frame
                elif ix <= 1 or ix >= self.NX-2 or iy <= 1 or iy >= self.NY-2:
                    img[iy,ix] = [0.0, 0.0, 0.0]  # Black
                
                # Draw PML boundaries
                elif ((self.USE_PML_XMIN and ix == self.NPOINTS_PML) or
                      (self.USE_PML_XMAX and ix == self.NX - self.NPOINTS_PML) or
                      (self.USE_PML_YMIN and iy == self.NPOINTS_PML) or
                      (self.USE_PML_YMAX and iy == self.NY - self.NPOINTS_PML)):
                    img[iy,ix] = [1.0, 0.588, 0.0]  # Orange-yellow
                
                # Draw receivers
                elif any((ix == rx and iy == rz) for rx, rz in zip(self.rec_x, self.rec_z)):
                    img[iy,ix] = [0.0, 1.0, 0.0]  # Green
                
                # Values below threshold
                elif abs(image_data_2D[ix,iy]) <= max_amplitude * cutvect:
                    if WHITE_BACKGROUND:
                        img[iy,ix] = [1.0, 1.0, 1.0]  # White
                    else:
                        img[iy,ix] = [0.0, 0.0, 0.0]  # Black
                
                # Regular points
                else:
                    if normalized_value >= 0.0:
                        # Red for positive values
                        img[iy,ix] = [normalized_value**POWER_DISPLAY, 0.0, 0.0]
                    else:
                        # Blue for negative values
                        img[iy,ix] = [0.0, 0.0, abs(normalized_value)**POWER_DISPLAY]
        
        # Save the image
        plt.imsave(fig_name, img)

    def simulate(self):
        """Run the main simulation"""
        # Check Courant stability condition
        quasi_cp_max = cp.maximum(cp.sqrt(self.c22/self.rho), cp.sqrt(self.c11/self.rho))
        Courant_number = quasi_cp_max * self.DELTAT * cp.sqrt(1.0/self.DELTAX**2 + 1.0/self.DELTAY**2)
        print(f'Courant number is {float(Courant_number)}')
        if Courant_number > 1.0:
            raise ValueError('Time step is too large, simulation will be unstable')
        
        # Setup PML boundaries
        self.setup_pml_boundary()
        self.setup_pml_boundary_y()
        
        # Time stepping
        for it in range(1, self.NSTEP + 1):
            if it % 100 == 0:
                print(f'Processing step {it}/{self.NSTEP}...')
            
            # Compute stress sigma and update memory variables
            self.compute_stress()
            
            # Compute velocity and update memory variables
            self.compute_velocity()
            
            # Add source
            self.add_source(it)
            
            # Apply Dirichlet boundary conditions
            self.apply_boundary_conditions()
            
            # Record seismograms
            self.record_seismograms(it)
            
            # Output information
            if it % self.IT_DISPLAY == 0 or it == 5:
                self.output_info(it)
        
        # Plot seismograms at the end of simulation
        self.plot_seismograms()
        print("\nEnd of the simulation")

if __name__ == '__main__':
    simulator = SeismicCPML2DAniso()
    simulator.simulate()