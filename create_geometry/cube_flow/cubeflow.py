import numpy as np 
from physicsnemo.sym.geometry.primitives_3d import Box,Plane, Channel, Sphere, Cylinder
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
from physicsnemo.utils.mesh import sdf_to_stl

import os
import math
import warnings

import physicsnemo
from physicsnemo.sym.geometry.primitives_2d import Rectangle, Circle
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk
from physicsnemo.sym.geometry.parameterization import Parameterization, Parameter

from physicsnemo.sym.models.fully_connected import FullyConnectedArch

from sympy import Symbol, Eq, Abs

import physicsnemo.sym
from physicsnemo.sym.hydra import to_absolute_path, instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint
)

from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.domain.inferencer import PointwiseInferencer
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.basic import NormalDotVec
from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter
)

class Geometry:
    def __init__(self,cylinder_height:float = 1.0,nr_points:int = 10000):
        self.cylinder_height = cylinder_height
        self.nr_points = nr_points

        self.channel_origin = (0,0,0)
        self.channel_dim = (4,1,1)

        self.cylinder_base_origin = self._get_cyl_base(self.channel_origin,self.cylinder_height)

        self.box = Channel((channel_origin[0]-channel_dim[0],
                              channel_origin[1]-channel_dim[1],
                              channel_origin[2]-channel_dim[2]),
                              (channel_origin[0]+channel_dim[0],
                              channel_origin[1]+channel_dim[1],
                              channel_origin[2]+channel_dim[2]
                              )
                              )
        
        self.cylinder = Cylinder(center=self.cylinder_base_origin,radius = 0.5, height = self.cylinder_height)

        self.inlet = Plane(point_1=(-4,-1,-1),point_2=(-4,1,1),normal=1)

        self.exit = Plane(point_1=(4,-1,-1),point_2=(4,1,1),normal=-1)

        self.geo = self.box - self.cylinder



    @staticmethod
    def _get_cyl_base(origin,height):
        return (origin[0],origin[1],origin[2]+(1-height/2))
    


nr_points = 10000

channel_origin = (0,0,0)
channel_dim = (4,1,1)

cylinder_height = 1.0
cylinder_base_origin = (0,0,channel_origin[2]+(1-(cylinder_height/2)))


box = Channel((channel_origin[0]-channel_dim[0],
                              channel_origin[1]-channel_dim[1],
                              channel_origin[2]-channel_dim[2]
),(channel_origin[0]+channel_dim[0],
                              channel_origin[1]+channel_dim[1],
                              channel_origin[2]+channel_dim[2]
))


# s = inlet.sample_boundary(nr_points=nr_points)

# var_to_polyvtk(s,'inlet')

# s = exit.sample_boundary(nr_points=nr_points)
# var_to_polyvtk(s,'exit')

# s = geo.sample_interior(nr_points=nr_points)
# var_to_polyvtk(s,'domain')

# s = geo.sample_boundary(nr_points=nr_points)
# var_to_polyvtk(s,'domain_walls'

@physicsnemo.sym.main(config_path='conf',config_name='conf_flow')
def run(cfg: PhysicsNeMoConfig) -> None:
    ns = NavierStokes(nu=0.01,rho=1.0,dim=3,time=False)
    navier_stokes_nodes = ns.make_nodes()
    normal_dot_vel = NormalDotVec()

    input_keys = [Key('x'),Key('y'),Key('z')]
    flow_net = FullyConnectedArch(
        input_keys=input_keys,
        output_keys = [Key('u'),Key('v'),Key('w'),Key('p')]
    )

    flow_nodes = navier_stokes_nodes + normal_dot_vel.make_nodes() + [flow_net.make_node(name='flow_network')]

    geo = Geometry()

    inlet_vel = 1.0
    volumetric_flow = 1.0

    flow_domain = Domain()

    u_profile = inlet_vel * math.tanh((0.5-Abs(y))/0.02)*math.tanh((0.5-Abs(z))/0.02)

    constraint_inlet = PointwiseBoundaryConstraint(
        nodes = flow_nodes,
        geometry=geo.inlet,
        outvar = {'u':u_profile,'v':0,'w':0},
        batch_size=cfg.batch_size.Inlet,
        criteria=Eq(x,geo.channel_origin[0]),
        lambda_weighting={
            'u':1.0,
            'v':1.0,
            'w':1.0
        },
        batch_per_epoch=5000,
    )

    flow_domain.add_constraint(constraint_inlet,'inlet')

    # make solver
    slv = Solver(cfg, flow_domain)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
