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
def _simulate(image, percentage_crop):
    h, w, _ = image.shape
    h_crop, w_crop = int(h*percentage_crop / 2), int(w*percentage_crop / 2)
    cropped_image = image[h_crop:-h_crop, w_crop:-w_crop, :]
    return cropped_image.astype(np.uint8)

@attr.s(auto_attribs=True)
class NarrowFOVNoiseModelCPUImpl:
    percentage_crop: float

    def simulate(self, image):
        return _simulate(image, self.percentage_crop)


@registry.register_noise_model
@attr.s(auto_attribs=True, kw_only=True)
class NarrowFOVNoiseModel(SensorNoiseModel):
    percentage_crop: float = 0.2

    def __attrs_post_init__(self):
        self._impl = NarrowFOVNoiseModelCPUImpl(self.percentage_crop)

    @staticmethod
    def is_valid_sensor_type(sensor_type: SensorType) -> bool:
        return sensor_type == SensorType.COLOR

    def simulate(self, image):
        return self._impl.simulate(image)

    def apply(self, image):
        r"""Alias of `simulate()` to conform to base-class and expected API
        """
        return self.simulate(image)
