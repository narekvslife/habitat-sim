#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import attr
import numba
import numpy as np
import cv2

from habitat_sim.registry import registry
from habitat_sim.sensor import SensorType
from habitat_sim.sensors.noise_models.sensor_noise_model import SensorNoiseModel


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _simulate(image, crack):
    mask_1d = crack.mean(axis=2) < 50
    mask_3d = np.zeros(mask_1d.shape + (3,))
    mask_3d[:, :, 0] = mask_1d
    mask_3d[:, :, 1] = mask_1d
    mask_3d[:, :, 2] = mask_1d

    return np.where(mask_3d, crack, image).astype(np.uint8)

@attr.s(auto_attribs=True)
class CrackNoiseModelCPUImpl:
    crack: np.ndarray

    def simulate(self, image):
        return _simulate(image, self.crack)


@registry.register_noise_model
@attr.s(auto_attribs=True, kw_only=True)
class CrackNoiseModel(SensorNoiseModel):
    def __attrs_post_init__(self):
        self._impl = CrackNoiseModelCPUImpl(self.crack)
        self.crack = cv2.imread("data/crack.png")

    @staticmethod
    def is_valid_sensor_type(sensor_type: SensorType) -> bool:
        return sensor_type == SensorType.COLOR

    def simulate(self, image):
        return self._impl.simulate(image)

    def apply(self, image):
        r"""Alias of `simulate()` to conform to base-class and expected API
        """
        return self.simulate(image)
