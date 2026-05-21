import openseespy.opensees as ops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import json

class RCBuildingDatasetGenerator:
    """
    Generate RC Building Dataset for Seismic Analysis
    
    Dataset Properties:
    - 11 building types (2-12 stories)
    - 3 bay lengths (5.0m, 6.1m, 7.6m)
    - High seismic zone (SD1=0.6g, SDs=1.0g)
    - Soil type D
    - Target: Maximum Interstory Drift Ratio (IDRmax) and Sa(T1)
    """
    
    def __init__(self):
        # Material properties
        self.fc = 34.5  # MPa - concrete compressive strength
        self.fy = 420   # MPa - steel yield strength
        self.Es = 200000  # MPa - steel elastic modulus
        self.Ec = 4700 * np.sqrt(self.fc)  # MPa - concrete elastic modulus
        
        # Loading
        self.dead_load = 8.4  # kN/m²
        self.live_load = 2.4  # kN/m²
        
        # Seismic parameters
        self.SD1 = 0.6  # g
        self.SDs = 1.0  # g
        self.R = 8
        self.Cd = 5.5
        self.omega = 3
        
        # Bay lengths
        self.bay_lengths = [5.0, 6.1, 7.6]  # meters
        
        # Story heights
        self.story_height = 4.0  # meters (typical)
        
        # Building configurations
        self.building_configs = self._get_building_configs()
        
        self.dataset = []
        
    def _get_building_configs(self):
        """Define structural element dimensions for each building type"""
        configs = {
            2: {
                'columns': {'h': 0.60, 'b': 0.60, 's': 0.127, 'p_tot': 0.0133, 'p_sh': 0.01},
                'beams': {'h': 0.60, 'b': 0.60, 's': 0.127, 'p': 0.0065, 'p_prime': 0.0075, 'p_sh': 0.004}
            },
            3: {
                'columns': {'h': 0.762, 'b': 0.7112, 's': 0.127, 'p_tot': 0.0133, 'p_sh': 0.01},
                'beams': {'h': 0.7112, 'b': 0.7112, 's': 0.127, 'p': 0.0065, 'p_prime': 0.0075, 'p_sh': 0.004}
            },
            4: {
                'columns': {'h': 0.762, 'b': 0.813, 's': 0.089, 'p_tot': 0.02, 'p_sh': 0.0085},
                'beams': {'h': 0.61, 'b': 0.813, 's': 0.127, 'p': 0.0108, 'p_prime': 0.0123, 'p_sh': 0.005}
            },
            5: {
                'columns': {'h': 0.762, 'b': 0.813, 's': 0.089, 'p_tot': 0.021, 'p_sh': 0.0085},
                'beams': {'h': 0.61, 'b': 0.813, 's': 0.127, 'p': 0.0108, 'p_prime': 0.0123, 'p_sh': 0.005}
            },
            6: {
                'columns': {'h': 0.7112, 'b': 0.6604, 's': 0.0889, 'p_tot': 0.02, 'p_sh': 0.01},
                'beams': {'h': 0.762, 'b': 0.66, 's': 0.127, 'p': 0.007, 'p_prime': 0.0075, 'p_sh': 0.0042}
            },
            7: {
                'columns': {'h': 0.7112, 'b': 0.6604, 's': 0.0889, 'p_tot': 0.02, 'p_sh': 0.01},
                'beams': {'h': 0.762, 'b': 0.66, 's': 0.127, 'p': 0.007, 'p_prime': 0.0075, 'p_sh': 0.0042}
            },
            8: {
                'columns': {'h': 0.7112, 'b': 0.6604, 's': 0.0889, 'p_tot': 0.02, 'p_sh': 0.01},
                'beams': {'h': 0.762, 'b': 0.66, 's': 0.127, 'p': 0.007, 'p_prime': 0.0075, 'p_sh': 0.0042}
            },
            9: {
                'columns': {'h': 0.7112, 'b': 0.665, 's': 0.089, 'p_tot': 0.02, 'p_sh': 0.01},
                'beams': {'h': 0.762, 'b': 0.66, 's': 0.127, 'p': 0.007, 'p_prime': 0.0075, 'p_sh': 0.0042}
            },
            10: {
                'columns': {'h': 0.813, 'b': 0.7112, 's': 0.089, 'p_tot': 0.023, 'p_sh': 0.0075},
                'beams': {'h': 0.965, 'b': 0.7112, 's': 0.1525, 'p': 0.006, 'p_prime': 0.0066, 'p_sh': 0.0045}
            },
            11: {
                'columns': {'h': 0.813, 'b': 0.7112, 's': 0.089, 'p_tot': 0.023, 'p_sh': 0.0075},
                'beams': {'h': 0.965, 'b': 0.7112, 's': 0.1525, 'p': 0.006, 'p_prime': 0.0066, 'p_sh': 0.0045}
            },
            12: {
                'columns': {'h': 0.813, 'b': 0.7112, 's': 0.089, 'p_tot': 0.023, 'p_sh': 0.0075},
                'beams': {'h': 0.965, 'b': 0.7112, 's': 0.1525, 'p': 0.006, 'p_prime': 0.0066, 'p_sh': 0.0045}
            }
        }
        return configs
    
    def create_opensees_model(self, n_stories, bay_length):
        """Create OpenSees model for given configuration"""
        # Clear any existing model
        ops.wipe()
        ops.reset()
        
        # Create model
        ops.model('basic', '-ndm', 2, '-ndf', 3)
        
        # Get building configuration
        config = self.building_configs[n_stories]
        
        # Define materials
        self._define_materials()
        
        # Create nodes
        node_coords = self._create_nodes(n_stories, bay_length)
        
        # Create elements
        self._create_elements(n_stories, bay_length, config)
        
        # Apply boundary conditions
        self._apply_boundary_conditions(bay_length)
        
        # Apply loads
        self._apply_loads(n_stories, bay_length)
        
        return node_coords
    
    def _define_materials(self):
        """Define concrete and steel materials with nonlinear properties"""
        # Concrete material (Concrete02)
        ops.uniaxialMaterial('Concrete02', 1, -self.fc, -0.002, -0.1*self.fc, -0.006, 0.1, 0.1*self.fc, 1000)
        
        # Steel material (Steel02)
        ops.uniaxialMaterial('Steel02', 2, self.fy, self.Es, 0.01, 18, 0.925, 0.15)
        
        # Fiber section for columns and beams will be defined in create_elements
    
    def _create_nodes(self, n_stories, bay_length):
        """Create nodes for the building frame"""
        node_coords = {}
        node_id = 1
        
        # Create nodes for each floor level
        for floor in range(n_stories + 1):
            y = floor * self.story_height
            for bay in range(4):  # 3 bays = 4 columns
                x = bay * bay_length
                ops.node(node_id, x, y)
                node_coords[node_id] = (x, y)
                node_id += 1
        
        return node_coords
    
    def _create_elements(self, n_stories, bay_length, config):
        """Create beam and column elements"""
        element_id = 1
        
        # Use a unique transformation tag for each building
        transf_tag = n_stories * 100 + int(bay_length * 10)
        
        # Use PDelta transformation
        ops.geomTransf('PDelta', transf_tag)

        # Column properties
        col_h = config['columns']['h']
        col_b = config['columns']['b']
        col_rho = config['columns']['p_tot']
        
        # Beam properties
        beam_h = config['beams']['h']
        beam_b = config['beams']['b']
        beam_rho = config['beams']['p']
        
        # Create fiber sections
        self._create_fiber_section(1, col_h, col_b, col_rho)  # Column section
        self._create_fiber_section(2, beam_h, beam_b, beam_rho)  # Beam section
        
        # Define coordinate transformation (added this line)
        ops.geomTransf('Linear', 1)  # Linear transformation for 2D frame


        # Create columns
        for floor in range(n_stories):
            for bay in range(4):
                node_i = floor * 4 + bay + 1
                node_j = (floor + 1) * 4 + bay + 1
                ops.element('nonlinearBeamColumn', element_id, node_i, node_j, 5, 1, transf_tag)
                element_id += 1
        
        # Create beams
        for floor in range(1, n_stories + 1):
            for bay in range(3):
                node_i = floor * 4 + bay + 1
                node_j = floor * 4 + bay + 2
                ops.element('nonlinearBeamColumn', element_id, node_i, node_j, 5, 2, transf_tag)
                element_id += 1
    
    def _create_fiber_section(self, sec_id, h, b, rho):
        """Create fiber section for RC elements"""
        # Create fiber section
        ops.section('Fiber', sec_id)
        
        # Define patch for concrete core
        ops.patch('rect', 1, 20, 20, -h/2, -b/2, h/2, b/2)
        
        # Define reinforcement layers
        As = rho * h * b / 4  # Area of steel per layer
        ops.layer('straight', 2, 2, As, -h/2+0.05, -b/2+0.05, -h/2+0.05, b/2-0.05)
        ops.layer('straight', 2, 2, As, h/2-0.05, -b/2+0.05, h/2-0.05, b/2-0.05)
    
    def _apply_boundary_conditions(self, bay_length):
        """Apply boundary conditions (fixed base)"""
        # Fix base nodes
        for bay in range(4):
            node_id = bay + 1
            ops.fix(node_id, 1, 1, 1)
    
    def _apply_loads(self, n_stories, bay_length):
        """Apply gravity loads"""

        # Create time series and load pattern
        ops.timeSeries('Constant', 1)
        ops.pattern('Plain', 1, 1)  # Create load pattern ID 1
        tributary_area = bay_length * bay_length
        total_load = (self.dead_load + 0.25 * self.live_load) * tributary_area
        
        # Apply loads to each floor
        for floor in range(1, n_stories + 1):
            for bay in range(4):
                node_id = floor * 4 + bay + 1
                # Interior vs exterior node load distribution
                if bay == 0 or bay == 3:  # Exterior nodes
                    load = total_load * 0.5
                else:  # Interior nodes
                    load = total_load
                ops.load(node_id, 0, -load, 0)
    
    def run_analysis(self, n_stories, bay_length):
        """Run static and dynamic analysis"""
        # Create model
        node_coords = self.create_opensees_model(n_stories, bay_length)
        
        # Static analysis for gravity loads
        ops.constraints('Plain')
        ops.numberer('Plain')
        ops.system('BandGeneral')
        ops.test('NormDispIncr', 1.0e-8, 6)
        ops.algorithm('Newton')
        ops.integrator('LoadControl', 0.1)
        ops.analysis('Static')
        ops.analyze(10)
        
        # Add this line to maintain constant gravity loads
        ops.loadConst('-time', 0.0)

        # Get fundamental period
        ops.wipeAnalysis()
        eigenvals = ops.eigen('-fullGenLapack', 1)
        T1 = 2 * np.pi / np.sqrt(eigenvals[0])
        
        # Calculate Sa(T1) from design spectrum
        if T1 <= 0.75 * self.SDs / self.SD1:
            Sa_T1 = self.SDs * (0.4 + 0.6 * T1 / (0.75 * self.SDs / self.SD1))
        elif T1 <= 1.5 * self.SDs / self.SD1:
            Sa_T1 = self.SDs
        else:
            Sa_T1 = self.SD1 / T1
        
        # Simplified IDR calculation (would require full dynamic analysis for accuracy)
        # This is a simplified estimation based on first mode response
        IDRmax = (Sa_T1 * T1**2 * self.Cd) / (4 * np.pi**2 * self.R * self.story_height)
        IDRmax = min(max(IDRmax, 0.002), 0.06)  # Bound within observed range
        
        return {
            'n_stories': n_stories,
            'bay_length': bay_length,
            'T1': T1,
            'Sa_T1': Sa_T1,
            'IDRmax': IDRmax,
            'total_height': n_stories * self.story_height,
            'column_dimensions': f"{self.building_configs[n_stories]['columns']['h']:.3f}x{self.building_configs[n_stories]['columns']['b']:.3f}",
            'beam_dimensions': f"{self.building_configs[n_stories]['beams']['h']:.3f}x{self.building_configs[n_stories]['beams']['b']:.3f}",
            'column_rho': self.building_configs[n_stories]['columns']['p_tot'],
            'beam_rho': self.building_configs[n_stories]['beams']['p']
        }
    
    def generate_dataset(self):
        """Generate complete dataset"""
        print("Generating RC Building Dataset...")
        
        # Generate all combinations
        story_range = range(2, 13)  # 2 to 12 stories
        
        for n_stories in story_range:
            for bay_length in self.bay_lengths:
                print(f"Processing {n_stories}-story building with {bay_length}m bays...")
                
                try:
                    result = self.run_analysis(n_stories, bay_length)
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
        # Features: [x_coord, y_coord, node_type, axial_stiffness, rotational_stiffness]
        node_features = []
        node_coords = {}
        node_id = 0
        
        for floor in range(n_stories + 1):
            for bay in range(4):
                x = bay * bay_length
                y = floor * self.story_height
                
                # Node type: 0=corner, 1=edge, 2=interior, 3=base
                if floor == 0:
                    node_type = 3  # Base
                elif bay == 0 or bay == 3:
                    node_type = 0  # Corner
                else:
                    node_type = 1 if floor == n_stories else 2  # Edge or Interior
                
                # Simplified stiffness values
                axial_stiff = 50000 if floor == 0 else 20000
                rot_stiff = 8000 if node_type == 2 else 1000
                
                node_features.append([x, y, node_type, axial_stiff, rot_stiff])
                node_coords[node_id] = (x, y)
                node_id += 1
        
        # Edge features and connectivity
        # Edges represent beams and columns
        edge_index = []
        edge_features = []
        
        # Column connections (vertical)
        for floor in range(n_stories):
            for bay in range(4):
                node_i = floor * 4 + bay
                node_j = (floor + 1) * 4 + bay
                edge_index.append([node_i, node_j])
                edge_index.append([node_j, node_i])  # Undirected graph
                
                # Column features: [element_type, length, cross_section_area, moment_capacity]
                col_area = config['columns']['h'] * config['columns']['b']
                col_moment = col_area * config['columns']['h'] * self.fc * 0.1  # Simplified
                
                edge_features.extend([
                    [0, self.story_height, col_area, col_moment],  # Column
                    [0, self.story_height, col_area, col_moment]   # Reverse direction
                ])
        
        # Beam connections (horizontal)
        for floor in range(1, n_stories + 1):
            for bay in range(3):
                node_i = floor * 4 + bay
                node_j = floor * 4 + bay + 1
                edge_index.append([node_i, node_j])
                edge_index.append([node_j, node_i])  # Undirected graph
                
                # Beam features
                beam_area = config['beams']['h'] * config['beams']['b']
                beam_moment = beam_area * config['beams']['h'] * self.fc * 0.08  # Simplified
                
                edge_features.extend([
                    [1, bay_length, beam_area, beam_moment],  # Beam
                    [1, bay_length, beam_area, beam_moment]   # Reverse direction
                ])
        
        return np.array(node_features), np.array(edge_index).T, np.array(edge_features)

# Usage example
if __name__ == "__main__":
    # Generate dataset
    generator = RCBuildingDatasetGenerator()
    
    # Generate single building example
    print("=" * 50)
    print("RC BUILDING DATASET GENERATION")
    print("=" * 50)
    
    # Example: 5-story building with 6.1m bays
    example_result = generator.run_analysis(5, 6.1)
    print(f"Example 5-story building results:")
    for key, value in example_result.items():
        print(f"  {key}: {value}")
    
    # Generate graph representation
    node_feat, edge_idx, edge_feat = generator.get_graph_representation(5, 6.1)
    print(f"\nGraph representation:")
    print(f"  Node features shape: {node_feat.shape}")
    print(f"  Edge index shape: {edge_idx.shape}")
    print(f"  Edge features shape: {edge_feat.shape}")
    
    # Generate full dataset (commented out for demo)
    # df = generator.generate_dataset()
    # generator.save_dataset()
    # print(f"Generated dataset with {len(df)} samples")