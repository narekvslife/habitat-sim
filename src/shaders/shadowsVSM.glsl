// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#if defined(SHADOWS_VSM)
precision highp float;
uniform samplerCube shadowMap0;
const float vsmBias = 0.16f;

float computeShadowUpperBound(vec2 moments, float fragLinearDepth) {
  // eliminates cubemap boundary thin line
  // p == 0 if moments.x < fragLinearDepth;
  // p == 1 otherwise
  // reference:
  // https://github.com/sydneyzh/variance_shadow_mapping_vk/blob/gh-vsm-master/shadow/onscreen.frag
  float p = step(fragLinearDepth, moments.x + vsmBias);

  float variance = max(moments.y - moments.x * moments.x, 0.0001);
  float d = fragLinearDepth - moments.x;
  float pMax = variance / (variance + d * d);

  return max(p, pMax);
}

float computeShadowVSM(vec3 fragPos, vec3 lightPos) {
  vec3 lightToFrag = fragPos - lightPos;
  float d = length(lightToFrag);

  vec2 moments = texture(shadowMap0, normalize(lightToFrag)).xy;
    // moments.x is the mean value while moments.y equals to depth * depth
    return computeShadowUpperBound(moments, d);
}

#endif