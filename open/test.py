import openseespy.opensees as ops
import numpy as np
import torch
from torch_geometric.data import Data

# =================================================================
# Configuration (Customize for different buildings)
# =================================================================
config = {
    # Structural dimensions
    "num_stories": 4,
    "bay_lengths_x": [5.0, 6.1, 7.6],  # Asymmetric X-direction
    "bay_lengths_y": [6.1, 6.1],       # Symmetric Y-direction
    "story_height": 3.0,
    "cantilever_offset": 1.2,
    
    # Material properties
    "concrete_fc": 34.5,    # MPa
    "steel_fy": 420,        # MPa
    "poissons_ratio": 0.2,
    
    # Loading
    "dead_load": 8.4,       # kN/m²
    "live_load": 2.4,       # kN/m²
    
    # Seismic parameters
    "SD1": 0.6,
    "SDs": 1.0,
    "R": 8,
    "Cd": 5.5
}

# =================================================================
# 3D Asymmetric Model Builder
# =================================================================
def build_3d_building(config):
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)
    
    # Generate grid coordinates with asymmetry
    x_coords = np.cumsum([0] + config["bay_lengths_x"])
    y_coords = np.cumsum([0] + config["bay_lengths_y"])
    z_coords = np.arange(0, (config["num_stories"]+1)*config["story_height"], 
                        config["story_height"])
    
    # Create nodes with cantilever asymmetry
    node_id = 0
    nodes = []
    for zi, z in enumerate(z_coords):
        for yi, y in enumerate(y_coords):
            for xi, x in enumerate(x_coords):
                # Add cantilever offset
                if xi == len(x_coords)-1 and zi > 0:
                    x += config["cantilever_offset"]
                ops.node(node_id, x, y, z)
                nodes.append({
                    'id': node_id,
                    'x': x, 'y': y, 'z': z,
                    'is_cantilever': 1 if (xi == len(x_coords)-1 and 
                                         x > x_coords[-1]) else 0
                })
                node_id += 1
                
    # Material properties
    E_concrete = 4700 * np.sqrt(config["concrete_fc"])
    G_concrete = E_concrete / (2 * (1 + config["poissons_ratio"]))
    ops.nDMaterial('ElasticIsotropic', 1, E_concrete, config["poissons_ratio"])
    
    # Create columns
    col_id = 0
    for zi in range(len(z_coords)-1):
        for yi in range(len(y_coords)):
            for xi in range(len(x_coords)):
                bot_node = (zi)*len(y_coords)*len(x_coords) + yi*len(x_coords) + xi
                top_node = (zi+1)*len(y_coords)*len(x_coords) + yi*len(x_coords) + xi
                
                # Vary column size based on position
                if xi == 0 or xi == len(x_coords)-1:
                    A, Iz, Iy = 0.4*0.4, 0.4**4/12, 0.4**4/12
                else:
                    A, Iz, Iy = 0.3*0.3, 0.3**4/12, 0.3**4/12
                J = 2.25 * Iz
                
                ops.element('elasticBeamColumn', col_id, bot_node, top_node,
                           A, E_concrete, G_concrete, J, Iy, Iz, 
                           '-orient', 0, 0, 1)
                col_id += 1
                
    # Create beams with torsion
    beam_id = 1000
    for zi in range(1, len(z_coords)):
        for yi in range(len(y_coords)):
            for xi in range(len(x_coords)-1):
                left_node = (zi)*len(y_coords)*len(x_coords) + yi*len(x_coords) + xi
                right_node = left_node + 1
                L = x_coords[xi+1] - x_coords[xi]
                
                # Torsional properties
                A = 0.3*0.5 if yi == 0 else 0.25*0.4
                Iz = 0.3*0.5**3/12 if yi == 0 else 0.25*0.4**3/12
                Iy = 0.5*0.3**3/12 if yi == 0 else 0.4*0.25**3/12
                J = 0.001
                
                ops.element('elasticBeamColumn', beam_id, left_node, right_node,
                           A, E_concrete, G_concrete, J, Iy, Iz,
                           '-orient', 1, 0, 0)
                beam_id += 1
                
    # Create slabs with ShellMITC4
    slab_id = 2000
    slab_thickness = 0.2
    for zi in range(1, len(z_coords)):
        for yi in range(len(y_coords)-1):
            for xi in range(len(x_coords)-1):
                nodes = [
                    (zi)*len(y_coords)*len(x_coords) + yi*len(x_coords) + xi,
                    (zi)*len(y_coords)*len(x_coords) + yi*len(x_coords) + xi+1,
                    (zi)*len(y_coords)*len(x_coords) + (yi+1)*len(x_coords) + xi+1,
                    (zi)*len(y_coords)*len(x_coords) + (yi+1)*len(x_coords) + xi
                ]
                ops.element('ShellMITC4', slab_id, *nodes, 1, slab_thickness)
                slab_id += 1
                
    # Apply loads and boundary conditions
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for node in nodes:
        if node['z'] > 0:
            ops.load(node['id'], 0, 0, -config["dead_load"]*25, 0, 0, 0)
    ops.fixZ(0, 1, 1, 1, 1, 1, 1)  # Fix base

    # Run static analysis
    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Transformation')
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1.0)
    ops.analysis('Static')
    ops.analyze(1)

# =================================================================
# Feature Extraction and Graph Construction
# =================================================================
def create_graph_data(config):
    # Build model and run analysis
    build_3d_building(config)
    
    # Node features: [x, y, z, A, Iz, Iy, J, load_z, is_cantilever]
    node_features = []
    for n in ops.getNodeTags():
        x, y, z = ops.nodeCoord(n)
        load_z = ops.nodeReaction(n)[2]
        node_features.append([
            x, y, z,
            0.3*0.3 if x < max(config["bay_lengths_x"]) else 0.4*0.4,  # A
            0.3**4/12 if x < max(config["bay_lengths_x"]) else 0.4**4/12,  # Iz
            0.3**4/12 if x < max(config["bay_lengths_x"]) else 0.4**4/12,  # Iy
            2.25*0.3**4/12 if x < max(config["bay_lengths_x"]) else 2.25*0.4**4/12, # J
            abs(load_z),
            1 if x > max(config["bay_lengths_x"]) else 0
        ])
    
    # Edge features: [element_type, L, A, Iz, Iy, J]
    edge_indices = []
    edge_attrs = []
    for elem in ops.getEleTags():
        n1, n2 = ops.eleNodes(elem)
        edge_indices.append([n1, n2])
        
        elem_type = 0 if elem < 1000 else 1 if elem < 2000 else 2
        L = ops.eleResponse(elem, 'length')
        A = ops.eleResponse(elem, 'A')
        Iy = ops.eleResponse(elem, 'Iy')
        Iz = ops.eleResponse(elem, 'Iz')
        J = ops.eleResponse(elem, 'J')
        
        edge_attrs.append([elem_type, L, A, Iy, Iz, J])
    
    # Global features
    global_features = [
        config["num_stories"],
        np.mean(config["bay_lengths_x"]),
        config["SD1"],
        config["R"],
        config["cantilever_offset"]
    ]
    
    # Convert to PyTorch Geometric format
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    u = torch.tensor([global_features], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u)

# =================================================================
# Main Execution
# =================================================================
if __name__ == "__main__":
    # Generate graph data
    building_data = create_graph_data(config)
    
    # Save dataset
    torch.save(building_data, '3d_asymmetric_building.pt')
    print(f'Generated graph with {building_data.num_nodes} nodes and {building_data.num_edges} edges')
    print(f'Node features: {building_data.x.shape}, Edge features: {building_data.edge_attr.shape}')