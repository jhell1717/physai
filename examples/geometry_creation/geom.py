import os
import warnings

import torch
import numpy as np
from sympy import Symbol, sqrt, Max

import physicsnemo.sym 
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint
)

from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.monitor import PointwiseMonitor
from physicsnemo.sym.key import Key

from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.utils.io import csv_to_dict
from physicsnemo.sym.geometry.tessellation import Tessellation



@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    point_path = to_absolute_path('./stl_files')
    inlet_mesh = Tessellation.from_stl(point_path+'/aneurysm_inlet.stl',airtight=False)
    outlet_mesh = Tessellation.from_stl(point_path+'/aneurysm_outlet.stl',airtight=False)
    noslip_mesh = Tessellation.from_stl(point_path+'/aneurysm_noslip.st',airtight=False)
    integral_mesh = Tessellation.from_stl(point_path+'/aneurysm_integral.stl',airtight=False)
    interior_mesh = Tessellation.from_stl(point_path+'/aneurysm_closed.stl',airtight=True)

    nu = 0.025
    inlet_vel = 1.5

    def circular_parabola(x,y,z,center,normal,radius,max_vel):
        centered_x = x - center[0]
        centered_y = y - center[1]
        centered_z = z - center[2]
        distance = sqrt(centered_x**2 + centered_y**2 + centered_z**2)
        parabola = max_vel * Max((1-(distance/radius)**2),0)
        return normal[0] * parabola, normal[1] * parabola, normal[2] * parabola
    
    def normalise_mesh(mesh,center,scale):
        mesh = mesh.translate([-c for c in center])
        mesh = mesh.scale(scale)
        return mesh
    
    def normalise_invar(invar, center,scale, dims =2):
        invar['x'] -= center[0]
        invar['y'] -= center[1]
        invar['z'] -= center[2]

        invar['x'] *= scale
        invar['y'] *= scale
        invar['z'] *= scale

        if 'area' in invar.keys():
            invar['area'] *= scale**dims
        return invar
    
    # scale and normalize mesh and openfoam data
    center = (-18.40381048596882, -50.285383353981196, 12.848136936899031)
    scale = 0.4
    inlet_mesh = normalise_mesh(inlet_mesh, center, scale)
    outlet_mesh = normalise_mesh(outlet_mesh, center, scale)
    noslip_mesh = normalise_mesh(noslip_mesh, center, scale)

    # geom params
    inlet_normal = (0.8526, -0.428, 0.299)
    inlet_area = 21.1284 * (scale**2)
    inlet_center = (-4.24298030045776, 4.082857101816247, -4.637790193399717)
    inlet_radius = np.sqrt(inlet_area / np.pi)
    outlet_normal = (0.33179, 0.43424, 0.83747)
    outlet_area = 12.0773 * (scale**2)
    outlet_radius = np.sqrt(outlet_area / np.pi)

    domain = Domain()

    ns = NavierStokes(nu=nu*scale,rho=1.0,dim=3,time=False)
    normal_dot_vel = NormalDotVec(['u','v','w'])
    flow_net = instantiate_arch(
        input_keys = [Key('x'),Key('y'),Key('z')],
        output_keys = [Key('u'),Key('v'),Key('w'),Key('p')],
        cfd = cfg.arch.fully_connected

    )

    nodes = ns.make_nodes() + normal_dot_vel.make_nodes() + [flow_net.make_node(name='flow_network')]

    outlet_radius = np.sqrt(outlet_area/np.pi)

    u, v, w = circular_parabola(
        Symbol("x"),
        Symbol("y"),
        Symbol("z"),
        center=inlet_center,
        normal=inlet_normal,
        radius=inlet_radius,
        max_vel=inlet_vel,
    )

    inlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry=inlet_mesh,
        outvar={'u':u,'v':v,'w':w},
        batch_size = cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet,'inlet')

    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_mesh,
        outvar={'p':0.0},
        batch_size=cfg.batch_size.outlet,
    )

    domain.add_constraint(outlet,'outlet')

    no_slip = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=noslip_mesh,
        outvar = {'u':0.0,'v':0.0,'w':0.0}
        batch_size = cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip,'no_slip')

    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=interior_mesh,
        outvar={'continuity':0.0,'momentum_x':0.0,'momentum_y':0.0,'momentum_z':0.0},
        batch_size = cfg.batch_size.interior
    )

    domain.add_constraint(interior,'interior')

    integral_continuity = IntegralBoundaryConstraint(
        nodes = nodes, 
        geometry=outlet_mesh,
        outvar={'normal_dot_vel': 2.540},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={'normal_dot_vel':0.1}
    )

    domain.add_constraint(integral_continuity,'integral_continuity_1')

    integral_continuity = IntegralBoundaryConstraint(
        nodes = nodes,
        geometry=integral_mesh,
        outvar={'normal_dot_vel':-2.540},
        batch_size=1,
        integral_batch_size=cfg.batch_size.integral_continuity,
        lambda_weighting={'normal_dot_vel':0.1}
    )

    domain.add_constraint(integral_continuity,'integral_continuity_2')

    file_path = r'/home/jhell/Desktop/physai/examples/geometry_creation/openfoam/aneurysm_parabolicInlet_sol0.csv'
    if os.path.exists(to_absolute_path(file_path)):
        mapping = {
            'Points:0':'x',
            'Points:1':'y',
            'Points:2':'z',
            'U:0':'u',
            'U:1':'v',
            'U:2':'w',
            'p':'p'
        }
        openfoam_var = csv_to
    