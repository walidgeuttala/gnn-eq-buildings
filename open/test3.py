import openseespy.opensees as ops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import json
import os
import requests
import zipfile
from scipy.interpolate import interp1d

class RCBuildingDatasetGenerator:
    """
    Generate 3D RC Building Dataset for Seismic Analysis with IDA
    
    Dataset Properties:
    - 11 building types (2-12 stories)
    - 3 bay lengths (5.0m, 6.1m, 7.6m)
    - High seismic zone (SD1=0.6g, SDs=1.0g)
    - Soil type D
    - Target: Maximum Interstory Drift Ratio (IDRmax) and Sa(T1)
    - 3D modeling with location-based element properties
    """
    
    def __init__(self):
        # Material properties
        self.fc = 34.5  # MPa - concrete compressive strength
        self.fy = 420   # MPa - steel yield strength
        self.fy_trans = 500  # MPa - transverse reinforcement yield strength
        self.Es = 200000  # MPa - steel elastic modulus
        self.Ec = 4700 * np.sqrt(self.fc)  # MPa - concrete elastic modulus
        self.cover = 0.04  # m - concrete cover
        
        # Loading
        self.dead_load = 8.4  # kN/m²
        self.live_load = 2.4  # kN/m²
        self.live_load_reduction = 0.4  # ASCE 7-16 live load reduction factor
        
        # Seismic parameters
        self.SD1 = 0.6  # g
        self.SDs = 1.0  # g
        self.R = 8
        self.Cd = 5.5
        self.omega = 3
        
        # Building dimensions
        self.bay_lengths = [5.0, 6.1, 7.6]  # meters
        self.num_bays = 3  # 3x3 bay system
        self.story_height = 4.0  # meters (typical)
        
        # Analysis parameters
        self.ground_motions = self._load_ground_motions()
        self.ida_scales = np.linspace(0.1, 2.0, 10)  # IDA scale factors
        self.damping_ratio = 0.05  # 5% damping
        
        # Building configurations
        self.building_configs = self._get_building_configs()
        
        self.dataset = []
        
    def _load_ground_motions(self):
        """Load FEMA P-695 ground motion set"""
        # In a real implementation, you would load actual ground motion records
        # This is a placeholder that returns dummy data for demonstration
        return [
            {"name": "GM1", "dt": 0.01, "accel": np.random.normal(0, 0.3, 1000)},
            {"name": "GM2", "dt": 0.01, "accel": np.random.normal(0, 0.3, 1000)}
        ]
        
    def _get_building_configs(self):
        """Define structural element dimensions for each building type with location differentiation"""
        base_configs = {
            2: {
                'columns': {
                    'corner': {'h': 0.55, 'b': 0.55, 'p_tot': 0.012, 'p_sh': 0.008},
                    'edge': {'h': 0.60, 'b': 0.60, 'p_tot': 0.0133, 'p_sh': 0.01},
                    'interior': {'h': 0.65, 'b': 0.65, 'p_tot': 0.015, 'p_sh': 0.012}
                },
                'beams': {'h': 0.60, 'b': 0.60, 'p': 0.0065, 'p_prime': 0.0075, 'p_sh': 0.004}
            },
            3: {
                'columns': {
                    'corner': {'h': 0.70, 'b': 0.65, 'p_tot': 0.0125, 'p_sh': 0.0085},
                    'edge': {'h': 0.762, 'b': 0.7112, 'p_tot': 0.0133, 'p_sh': 0.01},
                    'interior': {'h': 0.80, 'b': 0.75, 'p_tot': 0.0155, 'p_sh': 0.0125}
                },
                'beams': {'h': 0.7112, 'b': 0.7112, 'p': 0.0065, 'p_prime': 0.0075, 'p_sh': 0.004}
            },
            # Additional configurations for 4-12 stories would follow similar pattern
            # For brevity, only 2 and 3 are shown here
            12: {
                'columns': {
                    'corner': {'h': 0.75, 'b': 0.70, 'p_tot': 0.020, 'p_sh': 0.010},
                    'edge': {'h': 0.813, 'b': 0.7112, 'p_tot': 0.023, 'p_sh': 0.0075},
                    'interior': {'h': 0.90, 'b': 0.80, 'p_tot': 0.026, 'p_sh': 0.015}
                },
                'beams': {'h': 0.965, 'b': 0.7112, 'p': 0.006, 'p_prime': 0.0066, 'p_sh': 0.0045}
            }
        }
        
        # Create full configs for 2-12 stories with size reduction
        configs = {}
        for stories in range(2, 13):
            if stories in base_configs:
                config = base_configs[stories]
            else:
                # Interpolate for missing stories
                lower = max(k for k in base_configs.keys() if k <= stories)
                upper = min(k for k in base_configs.keys() if k >= stories)
                ratio = (stories - lower) / (upper - lower)
                
                config = {
                    'columns': {},
                    'beams': {}
                }
                
                # Interpolate column properties
                for loc in ['corner', 'edge', 'interior']:
                    lower_col = base_configs[lower]['columns'][loc]
                    upper_col = base_configs[upper]['columns'][loc]
                    config['columns'][loc] = {
                        'h': lower_col['h'] + ratio * (upper_col['h'] - lower_col['h']),
                        'b': lower_col['b'] + ratio * (upper_col['b'] - lower_col['b']),
                        'p_tot': lower_col['p_tot'] + ratio * (upper_col['p_tot'] - lower_col['p_tot']),
                        'p_sh': lower_col['p_sh'] + ratio * (upper_col['p_sh'] - lower_col['p_sh'])
                    }
                
                # Interpolate beam properties
                lower_beam = base_configs[lower]['beams']
                upper_beam = base_configs[upper]['beams']
                config['beams'] = {
                    'h': lower_beam['h'] + ratio * (upper_beam['h'] - lower_beam['h']),
                    'b': lower_beam['b'] + ratio * (upper_beam['b'] - lower_beam['b']),
                    'p': lower_beam['p'] + ratio * (upper_beam['p'] - lower_beam['p']),
                    'p_prime': lower_beam['p_prime'] + ratio * (upper_beam['p_prime'] - lower_beam['p_prime']),
                    'p_sh': lower_beam['p_sh'] + ratio * (upper_beam['p_sh'] - lower_beam['p_sh'])
                }
            
            # Apply size reduction for upper floors
            config['floor_reduction'] = {}
            for floor in range(stories):
                reduction = 1.0 - 0.05 * floor  # 5% reduction per floor
                config['floor_reduction'][floor] = max(reduction, 0.7)  # Minimum 70% of base size
            
            configs[stories] = config
        
        return configs
    
    def _get_column_properties(self, n_stories, floor, location):
        """Get column properties with floor-based size reduction"""
        config = self.building_configs[n_stories]
        base_props = config['columns'][location]
        reduction = config['floor_reduction'].get(floor, 1.0)
        
        return {
            'h': base_props['h'] * reduction,
            'b': base_props['b'] * reduction,
            'p_tot': base_props['p_tot'],
            'p_sh': base_props['p_sh']
        }
    
    def create_opensees_model(self, n_stories, bay_length):
        """Create 3D OpenSees model for given configuration"""
        # Clear any existing model
        ops.wipe()
        ops.reset()
        
        # Create 3D model
        ops.model('basic', '-ndm', 3, '-ndf', 6)
        
        # Get building configuration
        config = self.building_configs[n_stories]
        
        # Define materials
        self._define_materials()
        
        # Create nodes
        node_coords = self._create_nodes(n_stories, bay_length)
        
        # Create elements
        self._create_elements(n_stories, bay_length, config)
        
        # Apply boundary conditions
        self._apply_boundary_conditions()
        
        # Apply loads
        self._apply_loads(n_stories, bay_length)
        
        # Define rigid diaphragm constraints
        self._define_diaphragms(n_stories, bay_length)
        
        return node_coords
    
    def _define_materials(self):
        """Define concrete and steel materials with nonlinear properties"""
        # Confined concrete (Mander model)
        fpc = -self.fc  # Compressive strength (negative)
        eps_c0 = -0.002  # Strain at maximum strength
        fpcu = -0.2 * self.fc  # Ultimate strength
        eps_u = -0.006  # Ultimate strain
        lambda_param = 0.1  # Ratio between unloading/loading slopes
        ft = 0.1 * self.fc  # Tensile strength
        Ets = 0.1 * self.Ec  # Tension softening stiffness
        
        ops.uniaxialMaterial('Concrete04', 1, fpc, eps_c0, fpcu, eps_u, 
                            lambda_param, ft, -0.0001, Ets)
        
        # Unconfined concrete (cover)
        ops.uniaxialMaterial('Concrete04', 2, 0.8*fpc, eps_c0, 0.8*fpcu, eps_u, 
                            lambda_param, 0.8*ft, -0.0001, Ets)
        
        # Steel material (Steel02 with isotropic hardening)
        ops.uniaxialMaterial('Steel02', 3, self.fy, self.Es, 0.01, 18, 0.925, 0.15)
        
        # Transverse reinforcement
        ops.uniaxialMaterial('Steel02', 4, self.fy_trans, self.Es, 0.01, 20, 0.925, 0.15)
    
    def _create_nodes(self, n_stories, bay_length):
        """Create nodes for 3D building frame"""
        node_coords = {}
        node_id = 1
        
        # Create nodes for each floor level in a grid
        for floor in range(n_stories + 1):
            z = floor * self.story_height
            for x_idx in range(self.num_bays + 1):
                x = x_idx * bay_length
                for y_idx in range(self.num_bays + 1):
                    y = y_idx * bay_length
                    ops.node(node_id, x, y, z)
                    node_coords[node_id] = (x, y, z)
                    node_id += 1
        
        return node_coords
    
    def _create_elements(self, n_stories, bay_length, config):
        """Create beam, column, and joint elements"""
        element_id = 1
        
        # Create columns
        for floor in range(n_stories):
            for x_idx in range(self.num_bays + 1):
                for y_idx in range(self.num_bays + 1):
                    # Determine column location type
                    if (x_idx == 0 or x_idx == self.num_bays) and (y_idx == 0 or y_idx == self.num_bays):
                        location = 'corner'
                    elif x_idx == 0 or x_idx == self.num_bays or y_idx == 0 or y_idx == self.num_bays:
                        location = 'edge'
                    else:
                        location = 'interior'
                    
                    # Get column properties with floor-based reduction
                    col_props = self._get_column_properties(n_stories, floor, location)
                    
                    # Create column section
                    sec_id = 1000 + element_id
                    self._create_column_fiber_section(sec_id, col_props)
                    
                    # Define geometric transformation
                    transf_tag = 1  # Use same transformation for all columns
                    if element_id == 1:
                        ops.geomTransf('PDelta', transf_tag, 0, 0, 1)
                    
                    # Node IDs
                    bottom_node = 1 + floor * (self.num_bays+1)**2 + x_idx*(self.num_bays+1) + y_idx
                    top_node = 1 + (floor+1) * (self.num_bays+1)**2 + x_idx*(self.num_bays+1) + y_idx
                    
                    # Create column element
                    ops.element('nonlinearBeamColumn', element_id, bottom_node, top_node, 
                               5, sec_id, transf_tag)
                    element_id += 1
        
        # Create beams in X direction
        beam_sec_id = 2000
        for floor in range(1, n_stories + 1):
            for y_idx in range(self.num_bays + 1):
                for x_idx in range(self.num_bays):
                    # Create beam section
                    sec_id = beam_sec_id + element_id
                    self._create_beam_fiber_section(sec_id, config['beams'])
                    
                    # Define geometric transformation
                    transf_tag = 2
                    if element_id == 1:  # Only define once
                        ops.geomTransf('Linear', transf_tag, 0, 1, 0)
                    
                    # Node IDs
                    left_node = 1 + floor * (self.num_bays+1)**2 + x_idx*(self.num_bays+1) + y_idx
                    right_node = left_node + (self.num_bays+1)
                    
                    # Create beam element
                    ops.element('nonlinearBeamColumn', element_id, left_node, right_node, 
                               5, sec_id, transf_tag)
                    element_id += 1
        
        # Create beams in Y direction
        for floor in range(1, n_stories + 1):
            for x_idx in range(self.num_bays + 1):
                for y_idx in range(self.num_bays):
                    # Create beam section
                    sec_id = beam_sec_id + element_id
                    self._create_beam_fiber_section(sec_id, config['beams'])
                    
                    # Define geometric transformation
                    transf_tag = 3
                    if element_id == 1:  # Only define once
                        ops.geomTransf('Linear', transf_tag, 1, 0, 0)
                    
                    # Node IDs
                    bottom_node = 1 + floor * (self.num_bays+1)**2 + x_idx*(self.num_bays+1) + y_idx
                    top_node = bottom_node + 1
                    
                    # Create beam element
                    ops.element('nonlinearBeamColumn', element_id, bottom_node, top_node, 
                               5, sec_id, transf_tag)
                    element_id += 1
    
    def _create_column_fiber_section(self, sec_id, props):
        """Create fiber section for RC columns with confinement"""
        h = props['h']
        b = props['b']
        p_tot = props['p_tot']
        p_sh = props['p_sh']
        
        # Create fiber section
        ops.section('Fiber', sec_id)
        
        # Define confined core (subtract cover)
        h_core = h - 2*self.cover
        b_core = b - 2*self.cover
        
        # Core concrete (confined)
        ops.patch('rect', 1, 10, 10, 
                  -h_core/2, -b_core/2, h_core/2, b_core/2)
        
        # Cover concrete (unconfined)
        # Top and bottom
        ops.patch('rect', 2, 2, 10, 
                  -h/2, -b/2, -h/2 + self.cover, b/2)
        ops.patch('rect', 2, 2, 10, 
                  h/2 - self.cover, -b/2, h/2, b/2)
        # Sides
        ops.patch('rect', 2, 10, 2, 
                  -h/2 + self.cover, -b/2, h/2 - self.cover, -b/2 + self.cover)
        ops.patch('rect', 2, 10, 2, 
                  -h/2 + self.cover, b/2 - self.cover, h/2 - self.cover, b/2)
        
        # Longitudinal reinforcement
        As_total = p_tot * h * b
        num_bars = max(4, int(As_total / 0.0001))  # At least 4 bars
        As_bar = As_total / num_bars
        
        # Define reinforcement layers
        ops.layer('straight', 3, num_bars, As_bar, 
                  -h/2 + self.cover, -b/2 + self.cover, 
                  -h/2 + self.cover, b/2 - self.cover)  # Top layer
        ops.layer('straight', 3, num_bars, As_bar, 
                  h/2 - self.cover, -b/2 + self.cover, 
                  h/2 - self.cover, b/2 - self.cover)  # Bottom layer
        
        # Transverse reinforcement
        s = 0.1  # Spacing (m)
        Ash = p_sh * s * b  # Area of transverse reinforcement
        ops.layer('straight', 4, 2, Ash, 
                  -h/2 + self.cover, 0, h/2 - self.cover, 0)  # Vertical ties
        ops.layer('straight', 4, 2, Ash, 
                  0, -b/2 + self.cover, 0, b/2 - self.cover)  # Horizontal ties
    
    def _create_beam_fiber_section(self, sec_id, props):
        """Create fiber section for RC beams"""
        h = props['h']
        b = props['b']
        p = props['p']
        p_prime = props['p_prime']
        
        # Create fiber section
        ops.section('Fiber', sec_id)
        
        # Concrete section
        ops.patch('rect', 2, 10, 10, -h/2, -b/2, h/2, b/2)
        
        # Tension reinforcement
        As_tension = p * h * b
        num_bars_t = max(2, int(As_tension / 0.0001))  # At least 2 bars
        As_bar_t = As_tension / num_bars_t
        ops.layer('straight', 3, num_bars_t, As_bar_t, 
                  -h/2 + self.cover, -b/2 + self.cover, 
                  -h/2 + self.cover, b/2 - self.cover)  # Top layer
        
        # Compression reinforcement
        As_comp = p_prime * h * b
        num_bars_c = max(2, int(As_comp / 0.0001))  # At least 2 bars
        As_bar_c = As_comp / num_bars_c
        ops.layer('straight', 3, num_bars_c, As_bar_c, 
                  h/2 - self.cover, -b/2 + self.cover, 
                  h/2 - self.cover, b/2 - self.cover)  # Bottom layer
    
    def _define_diaphragms(self, n_stories, bay_length):
        """Define rigid diaphragm constraints at each floor level"""
        for floor in range(1, n_stories + 1):
            master_node = None
            slave_nodes = []
            
            # Find all nodes at this floor level
            for node_id, (x, y, z) in self.node_coords.items():
                if abs(z - floor * self.story_height) < 0.001:
                    if master_node is None:
                        master_node = node_id
                    else:
                        slave_nodes.append(node_id)
            
            # Constrain all slave nodes to master node
            if master_node and slave_nodes:
                ops.rigidDiaphragm(3, master_node, *slave_nodes)
    
    def _apply_boundary_conditions(self):
        """Apply fixed base conditions"""
        for node_id, (x, y, z) in self.node_coords.items():
            if abs(z) < 0.001:  # Base nodes
                ops.fix(node_id, 1, 1, 1, 1, 1, 1)  # Fixed in all DOFs
    
    def _apply_loads(self, n_stories, bay_length):
        """Apply gravity loads based on ASCE 7-16 load combinations"""
        # Create time series and load pattern
        ops.timeSeries('Constant', 1)
        ops.pattern('Plain', 1, 1)
        
        # Gravity load combination: 1.2D + 0.5L
        load_factor_dead = 1.2
        load_factor_live = 0.5
        
        for floor in range(1, n_stories + 1):
            for x_idx in range(1, self.num_bays):
                for y_idx in range(1, self.num_bays):
                    # Node at this location
                    node_id = 1 + floor * (self.num_bays+1)**2 + x_idx*(self.num_bays+1) + y_idx
                    
                    # Determine tributary area based on location
                    if x_idx == 0 or x_idx == self.num_bays:
                        trib_x = bay_length / 2
                    else:
                        trib_x = bay_length
                    
                    if y_idx == 0 or y_idx == self.num_bays:
                        trib_y = bay_length / 2
                    else:
                        trib_y = bay_length
                    
                    tributary_area = trib_x * trib_y
                    
                    # Calculate loads
                    dead_load = self.dead_load * tributary_area * load_factor_dead
                    live_load = self.live_load * tributary_area * load_factor_live * self.live_load_reduction
                    total_load = -(dead_load + live_load)  # Downward direction
                    
                    # Apply vertical load
                    ops.load(node_id, 0, 0, total_load, 0, 0, 0)
    
    def run_ida(self, n_stories, bay_length):
        """Perform Incremental Dynamic Analysis (IDA)"""
        # Create model and apply gravity loads
        self.node_coords = self.create_opensees_model(n_stories, bay_length)
        
        # Run gravity analysis
        self._run_gravity_analysis()
        
        # Get fundamental period
        T1 = self._get_fundamental_period()
        
        # Perform IDA for each ground motion
        ida_results = []
        for gm in self.ground_motions:
            gm_results = self._run_ground_motion_analysis(gm, T1)
            ida_results.append(gm_results)
        
        # Calculate median IDRmax and Sa(T1)
        idrmax_values = [r['idrmax'] for r in ida_results]
        sa_t1_values = [r['sa_t1'] for r in ida_results]
        
        median_idrmax = np.median(idrmax_values)
        median_sa_t1 = np.median(sa_t1_values)
        
        return {
            'n_stories': n_stories,
            'bay_length': bay_length,
            'T1': T1,
            'IDRmax': median_idrmax,
            'Sa_T1': median_sa_t1,
            'ida_results': ida_results
        }
    
    def _run_gravity_analysis(self):
        """Run static gravity analysis"""
        ops.constraints('Plain')
        ops.numberer('Plain')
        ops.system('BandGeneral')
        ops.test('NormDispIncr', 1.0e-6, 10)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', 0.1)
        ops.analysis('Static')
        ops.analyze(10)
        ops.loadConst('-time', 0.0)
    
    def _get_fundamental_period(self):
        """Perform eigenvalue analysis to get fundamental period"""
        ops.wipeAnalysis()
        num_eigen = 3
        eigenvals = ops.eigen(num_eigen)
        T1 = 2 * np.pi / np.sqrt(eigenvals[0])
        return T1
    
    def _run_ground_motion_analysis(self, gm, T1):
        """Run dynamic analysis for a ground motion at multiple scales"""
        # Setup analysis parameters
        dt_analysis = gm['dt']
        t_total = len(gm['accel']) * dt_analysis
        
        # Create analysis objects
        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('UmfPack')
        ops.test('NormDispIncr', 1.0e-6, 10)
        ops.algorithm('Newton')
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')
        
        # Setup recorders (for IDR calculation)
        ops.recorder('Node', '-file', 'node_disp.out', '-time', '-nodeRange', 1, 10000, '-dof', 1, 2, 3, 'disp')
        
        # Run analysis for each scale factor
        idrmax_results = []
        sa_t1_results = []
        
        for scale in self.ida_scales:
            # Apply ground motion
            accel_series = ops.timeSeries('Path', 100, '-dt', dt_analysis, 
                                         '-values', *list(scale * gm['accel']), 
                                         '-factor', 9.81)  # Convert g to m/s²
            ops.pattern('UniformExcitation', 200, 1, '-accel', 100)  # X-direction
            
            # Reset analysis
            ops.wipeAnalysis()
            ops.analysis('Transient')
            
            # Run dynamic analysis
            ok = ops.analyze(int(t_total/dt_analysis), dt_analysis)
            
            if ok != 0:
                # Analysis failed - collapse reached
                idrmax = 0.10  # Assume collapse at 10% drift
                break
            
            # Calculate maximum interstory drift
            idrmax = self._calculate_idrmax()
            idrmax_results.append(idrmax)
            
            # Calculate Sa(T1) for this scale
            sa_t1 = self._calculate_sa_t1(gm, scale, T1)
            sa_t1_results.append(sa_t1)
            
            # Check for collapse
            if idrmax > 0.10:
                break
        
        # Return IDA curve for this ground motion
        return {
            'gm_name': gm['name'],
            'scales': self.ida_scales[:len(idrmax_results)],
            'idrmax': idrmax_results,
            'sa_t1': sa_t1_results,
            'collapse_point': next((i for i, idr in enumerate(idrmax_results) if idr > 0.10), None)
        }
    
    def _calculate_idrmax(self):
        """Calculate maximum interstory drift ratio from analysis results"""
        # In a real implementation, you would process node displacements
        # Here we return a simplified value for demonstration
        return np.random.uniform(0.001, 0.05)
    
    def _calculate_sa_t1(self, gm, scale, T1):
        """Calculate spectral acceleration at fundamental period"""
        # In a real implementation, you would compute response spectrum
        # Here we return a simplified value for demonstration
        return scale * np.max(np.abs(gm['accel'])) * 1.5 / (T1 + 0.5)
    
    def generate_dataset(self):
        """Generate complete dataset with IDA"""
        print("Generating RC Building Dataset with IDA...")
        
        # Generate all combinations
        story_range = range(2, 13)  # 2 to 12 stories
        
        for n_stories in story_range:
            for bay_length in self.bay_lengths:
                print(f"Processing {n_stories}-story building with {bay_length}m bays...")
                
                try:
                    result = self.run_ida(n_stories, bay_length)
                    self.dataset.append(result)
                except Exception as e:
                    print(f"Error in {n_stories}-story, {bay_length}m bay: {e}")
                    continue
        
        return pd.DataFrame(self.dataset)
    
    def save_dataset(self, filename='rc_building_dataset.csv'):
        """Save dataset to CSV"""
        df = pd.DataFrame(self.dataset)
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        return df
    
    def get_graph_representation(self, n_stories, bay_length):
        """
        Generate graph representation for GNN
        Returns: node_features, edge_index, edge_features
        """
        config = self.building_configs[n_stories]
        
        # Node features (each node represents a joint/connection point)
        # Features: [x, y, z, node_type, floor]
        node_features = []
        node_id = 0
        
        for floor in range(n_stories + 1):
            z = floor * self.story_height
            for x_idx in range(self.num_bays + 1):
                x = x_idx * bay_length
                for y_idx in range(self.num_bays + 1):
                    y = y_idx * bay_length
                    
                    # Node type: 0=corner, 1=edge, 2=interior, 3=base
                    if floor == 0:
                        node_type = 3  # Base
                    elif (x_idx == 0 or x_idx == self.num_bays) and (y_idx == 0 or y_idx == self.num_bays):
                        node_type = 0  # Corner
                    elif x_idx == 0 or x_idx == self.num_bays or y_idx == 0 or y_idx == self.num_bays:
                        node_type = 1  # Edge
                    else:
                        node_type = 2  # Interior
                    
                    node_features.append([x, y, z, node_type, floor])
                    node_id += 1
        
        # Edge features and connectivity
        edge_index = []
        edge_features = []
        edge_id = 0
        
        # Column connections (vertical)
        for floor in range(n_stories):
            for x_idx in range(self.num_bays + 1):
                for y_idx in range(self.num_bays + 1):
                    # Get column properties
                    if (x_idx == 0 or x_idx == self.num_bays) and (y_idx == 0 or y_idx == self.num_bays):
                        location = 'corner'
                    elif x_idx == 0 or x_idx == self.num_bays or y_idx == 0 or y_idx == self.num_bays:
                        location = 'edge'
                    else:
                        location = 'interior'
                    
                    col_props = self._get_column_properties(n_stories, floor, location)
                    
                    # Node indices
                    bottom_node = floor * (self.num_bays+1)**2 + x_idx*(self.num_bays+1) + y_idx
                    top_node = (floor+1) * (self.num_bays+1)**2 + x_idx*(self.num_bays+1) + y_idx
                    
                    edge_index.append([bottom_node, top_node])
                    
                    # Column features: [element_type, length, h, b, p_tot]
                    edge_features.append([
                        0,  # Column
                        self.story_height,
                        col_props['h'],
                        col_props['b'],
                        col_props['p_tot']
                    ])
                    edge_id += 1
        
        # Beam connections in X direction
        for floor in range(1, n_stories + 1):
            for y_idx in range(self.num_bays + 1):
                for x_idx in range(self.num_bays):
                    # Node indices
                    left_node = floor * (self.num_bays+1)**2 + x_idx*(self.num_bays+1) + y_idx
                    right_node = left_node + (self.num_bays+1)
                    
                    edge_index.append([left_node, right_node])
                    
                    # Beam features: [element_type, length, h, b, p]
                    edge_features.append([
                        1,  # Beam in X direction
                        bay_length,
                        config['beams']['h'],
                        config['beams']['b'],
                        config['beams']['p']
                    ])
                    edge_id += 1
        
        # Beam connections in Y direction
        for floor in range(1, n_stories + 1):
            for x_idx in range(self.num_bays + 1):
                for y_idx in range(self.num_bays):
                    # Node indices
                    bottom_node = floor * (self.num_bays+1)**2 + x_idx*(self.num_bays+1) + y_idx
                    top_node = bottom_node + 1
                    
                    edge_index.append([bottom_node, top_node])
                    
                    # Beam features: [element_type, length, h, b, p]
                    edge_features.append([
                        2,  # Beam in Y direction
                        bay_length,
                        config['beams']['h'],
                        config['beams']['b'],
                        config['beams']['p']
                    ])
                    edge_id += 1
        
        return np.array(node_features), np.array(edge_index).T, np.array(edge_features)

# Usage example
if __name__ == "__main__":
    # Generate dataset
    generator = RCBuildingDatasetGenerator()
    
    print("=" * 50)
    print("3D RC BUILDING DATASET GENERATION WITH IDA")
    print("=" * 50)
    
    # Example: 5-story building with 6.1m bays
    print("Running IDA for 5-story building with 6.1m bays...")
    ida_result = generator.run_ida(5, 6.1)
    
    print("\nIDA Results:")
    print(f"Fundamental Period (T1): {ida_result['T1']:.3f} s")
    print(f"Median IDRmax: {ida_result['IDRmax']:.4f}")
    print(f"Median Sa(T1): {ida_result['Sa_T1']:.3f} g")
    
    # Generate graph representation
    print("\nGenerating graph representation...")
    node_feat, edge_idx, edge_feat = generator.get_graph_representation(5, 6.1)
    print(f"  Node features shape: {node_feat.shape}")
    print(f"  Edge index shape: {edge_idx.shape}")
    print(f"  Edge features shape: {edge_feat.shape}")
    
    # Generate full dataset (commented out for performance)
    # print("\nGenerating full dataset...")
    # df = generator.generate_dataset()
    # generator.save_dataset()
    # print(f"Generated dataset with {len(df)} samples")