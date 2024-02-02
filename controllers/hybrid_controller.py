import numpy as np
import torch
from DATT.utils.math_utils import *
from DATT.controllers.mppi_controller import MPPIController
from DATT.controllers.datt_controller import DATTController
from DATT.controllers.cntrl_config import MPPIConfig
from DATT.controllers.cntrl_config import DATTConfig
from DATT.quadsim.control import Controller
from DATT.quadsim.models import RBModel
from DATT.configuration.configuration import AllConfig

class HybridController(Controller):
    def __init__(self, config: AllConfig, mppi_config: MPPIConfig, datt_config: DATTConfig):
        super().__init__()
        self.mppi_controller = MPPIController(config, mppi_config)
        self.datt_controller = DATTController(config, datt_config)
        self.mppi_weight = 0.3
        self.datt_weight = 0.7
        #a metric to evaluate performance (eg:tracking error)
        self.performance_metric = 0.0

    def update_weights(self, state):
        #update the weights based on the state or performance metrics
        #based on learning i guess, for now a placeholder
        w=0
        if w>1:
            self.mppi_weight = 0.7
            self.datt_weight = 0.3
        else:
            self.mppi_weight = 0.4
            self.datt_weight = 0.6
        #normalize
        total = self.mppi_weight + self.datt_weight
        self.mppi_weight /= total
        self.datt_weight /= total

    def response(self, **response_inputs):
        #control responses
        mppi_thrust, mppi_angvel = self.mppi_controller.response(**response_inputs)
        datt_thrust, datt_angvel = self.datt_controller.response(**response_inputs)
        mppi_thrust = torch.tensor(mppi_thrust, dtype=torch.float32)
        mppi_angvel = torch.tensor(mppi_angvel, dtype=torch.float32)
        datt_thrust = torch.tensor(datt_thrust, dtype=torch.float32)
        datt_angvel = torch.tensor(datt_angvel, dtype=torch.float32)
        #update the blending weights dynamically
        # state = response_inputs.get('state')
        # self.update_weights(state)
        #blend response from both
        blended_thrust = self.mppi_weight * mppi_thrust + self.datt_weight * datt_thrust
        blended_angvel = self.mppi_weight * mppi_angvel + self.datt_weight * datt_angvel
        # return blended_thrust, blended_angvel
        return blended_thrust.item(), blended_angvel.cpu().numpy()
