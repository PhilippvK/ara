// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Matteo Perotti

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "runtime.h"

#ifndef SPIKE
#include "printf.h"
#endif

#define N 640
#define M 128
// #define M 1

#include "muriscv_nn_functions.h"
#include "muriscv_nn_support_functions.h"

int main() {
  printf("\n");
  printf("=============\n");
  printf("=  .......  =\n");
  printf("=============\n");
  printf("\n");
  printf("\n");

  // Call the main kernel, and measure cycles
  start_timer();
  // const muriscv_nn_activation activation = {-128,127};
  // const muriscv_nn_fc_params fc_params = {-89, 0, -128, activation};
  // muriscv_nn_per_tensor_quant_params quant_params = {1638001653, -8};
  // muriscv_nn_dims input_dims = {1,1,1,N};
  // muriscv_nn_dims filter_dims = {N,1,1,M};
  // muriscv_nn_dims bias_dims = {1,1,1,M};
  // muriscv_nn_dims output_dims = {1,1,1,M};
  static q7_t input[N];
  static q7_t kernel[N*M];
  static int32_t bias[M];
  static q7_t output[M];
  q7_t *input_ptr = &input[0];
  q7_t *kernel_ptr = &kernel[0];
  int32_t *bias_ptr = &bias[0];
  volatile q7_t *output_ptr = &output[0];

  // int32_t batch_cnt = input_dims.n;
  int32_t batch_cnt = 1;

  while (batch_cnt) {
    const q7_t *lhs = input_ptr;
    const q7_t *rhs = kernel_ptr;
    const q31_t *bias = bias_ptr;
    q7_t *dst = output_ptr;
    // const int32_t lhs_offset = fc_params.input_offset;
    const int32_t lhs_offset = -89;
    // const int32_t rhs_offset = 0;
    // const int32_t dst_offset = fc_params.output_offset;
    const int32_t dst_offset = -128;
    // const int32_t dst_multiplier = quant_params.multiplier;
    const int32_t dst_multiplier = 1638001653;
    // const int32_t dst_shift = quant_params.shift;
    const int32_t dst_shift = -8;
    // const int32_t rhs_cols = filter_dims.n;
    const int32_t rhs_cols = N;
    // const int32_t rhs_rows = output_dims.n;
    const int32_t rhs_rows = 1;
    // const int32_t activation_min = fc_params.activation.min;
    const int32_t activation_min = -128;
    // const int32_t activation_max = fc_params.activation.max;
    const int32_t activation_max = 127;
    const int32_t address_offset = 1L;

    for (int i_loop_cnt = 0; i_loop_cnt < rhs_rows; i_loop_cnt++) {
        // printf("i_loop_cnt=%d\n", i_loop_cnt);
        const q7_t *lhs_ptr = &lhs[0];
        const q7_t *rhs_ptr = &rhs[0];

        size_t vl = vsetvl_e32m8(rhs_cols);
        vint32m8_t result = vmv_v_x_i32m8(0, vl);

        int32_t rhs_cols_cnt = rhs_cols;
        while (rhs_cols_cnt > 0) {
          // printf("rhs_cols_cnt=%d\n", rhs_cols_cnt);
          vl = vsetvl_e32m8(rhs_cols_cnt);
          // printf("A\n");

          vint32m8_t rhs_value = vsext_vf4_i32m8(vle8_v_i8m2(rhs_ptr, vl), vl);
          // printf("A\n");

          vint32m8_t lhs_value = vsext_vf4_i32m8(vle8_v_i8m2(lhs_ptr, vl), vl);
          // printf("A\n");
          lhs_value = vadd_vx_i32m8(lhs_value, lhs_offset, vl);
          // printf("A\n");

          result = vmacc_vv_i32m8(result, lhs_value, rhs_value, vl);
          // printf("A\n");

          rhs_ptr += vl;
          // printf("A\n");
          lhs_ptr += vl;
          // printf("A\n");
          rhs_cols_cnt -= vl;
          // printf("A\n");
        }

        vl = vsetvl_e32m1(1); // TODO(fabianpedd): Is this really needed?
        vint32m1_t reduct = vundefined_i32m1();
        if (bias) {
          reduct = vle32_v_i32m1(bias++, vl);
        }
        else {
          reduct = vmv_v_x_i32m1(0, vl);
        }

        vl = vsetvl_e32m8(rhs_cols);
        reduct = vredsum_vs_i32m8_i32m1(reduct, result, reduct, vl);
        q31_t res_scalar = vmv_x_s_i32m1_i32(reduct);

        res_scalar = muriscv_nn_requantize(res_scalar, dst_multiplier, dst_shift);
        res_scalar += dst_offset;
        res_scalar = MAX(res_scalar, activation_min);
        res_scalar = MIN(res_scalar, activation_max);

        *dst = (q7_t)res_scalar;
        dst += address_offset;
        rhs += rhs_cols;
    }

    // input_ptr += filter_dims.n;
    input_ptr += N;
    // output_ptr += output_dims.c;
    output_ptr += M;
    batch_cnt--;
  }
  stop_timer();

  // Performance metrics
  int64_t runtime = get_timer();
  // float performance = 2.0 * F * F * M * N / runtime;
  // float utilization = 100 * performance / (2.0 * NR_LANES);

  printf("The execution took %d cycles.\n", runtime);
  // printf("The performance is %f OP/cycle (%f%% utilization).\n", performance,
  //        utilization);

  // // Verify correctness
  // printf("Verifying result...\n");
  // int error = verify_matrix(o, golden_o, M, N);
  // if (error != 0) {
  //   printf("Fail.\n");
  // } else {
  //   printf("Passed.\n");
  // }

  return 0;
}
