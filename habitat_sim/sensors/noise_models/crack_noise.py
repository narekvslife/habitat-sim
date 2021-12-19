#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import attr
import numba
import numpy as np

from habitat_sim.registry import registry
from habitat_sim.sensor import SensorType
from habitat_sim.sensors.noise_models.sensor_noise_model import SensorNoiseModel


@numba.jit(nopython=True, parallel=True, fastmath=True)

def _simulate(image, crack):
    mask = crack > 240
    return np.where(mask, image, crack).astype(np.uint8)

@attr.s(auto_attribs=True)
class CrackNoiseModelCPUImpl:
    crack: np.ndarray

    def simulate(self, image):
        return _simulate(image, self.crack)


@registry.register_noise_model
@attr.s(auto_attribs=True, kw_only=True)
class CrackNoiseModel(SensorNoiseModel):
    def __attrs_post_init__(self):
        self.crack = np.load("data/crack.npy")
        self._impl = CrackNoiseModelCPUImpl(self.crack)

    @staticmethod
    def is_valid_sensor_type(sensor_type: SensorType) -> bool:
        return sensor_type == SensorType.COLOR

    def simulate(self, image):
        return self._impl.simulate(image)

    def apply(self, image):
        r"""Alias of `simulate()` to conform to base-class and expected API
        """
        return self.simulate(image)
