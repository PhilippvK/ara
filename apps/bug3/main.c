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

#include "muriscv_nn_functions.h"
#include "muriscv_nn_support_functions.h"


// #define N 640
#define N 4096
// #define M 128
#define M 1

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
  // // muriscv_nn_dims bias_dims = {1,1,1,M};
  // muriscv_nn_dims output_dims = {1,1,1,M};
  // static q7_t input[N];
  // static q7_t kernel[N*M];
  // static int32_t bias[M];
  // static q7_t output[M];
  // q7_t *input_ptr = &input[0];
  // q7_t *kernel_ptr = &kernel[0];
  // int32_t *bias_ptr = &bias[0];
  // q7_t *output_ptr = &output[0];

  // int32_t batch_cnt = input_dims.n;

  // while (batch_cnt) {
  //   muriscv_nn_vec_mat_mult_t_s8(input_ptr,
  //                                kernel_ptr,
  //                                bias_ptr,
  //                                output_ptr,
  //                                fc_params.input_offset,
  //                                0,
  //                                fc_params.output_offset,
  //                                quant_params.multiplier,
  //                                quant_params.shift,
  //                                filter_dims.n, /* col_dim or accum_depth */
  //                                output_dims.c, /* row_dim or output_depth */
  //                                fc_params.activation.min,
  //                                fc_params.activation.max,
  //                                1L);
  //   input_ptr += filter_dims.n;
  //   output_ptr += output_dims.c;
  //   batch_cnt--;
  // }
  int8_t lhs[N];
  // int8_t rhs[N*M];
  // int8_t dst[M];
  size_t vl = vsetvl_e32m8(N);
  // vint32m8_t result = vmv_v_x_i32m8(0, vl);
  // vint32m8_t rhs_value = vsext_vf4_i32m8(vle8_v_i8m2(&rhs[0], vl), vl);
  vint32m8_t lhs_value = vsext_vf4_i32m8(vle8_v_i8m2(&lhs[0], vl), vl);
  // vint8m2_t lhs_value = vle8_v_i8m2(&lhs[0], vl);
  // lhs_value = vadd_vx_i32m8(lhs_value, 0, vl); // ?
  // result = vmacc_vv_i32m8(result, lhs_value, rhs_value, vl);
  // vl = vsetvl_e32m1(1);
  // vint32m1_t reduct = vundefined_i32m1();
  // reduct = vmv_v_x_i32m1(0, vl);
  // vint32m1_t reduct = vmv_v_x_i32m1(0, vl);
  // vl = vsetvl_e32m8(N);
  // reduct = vredsum_vs_i32m8_i32m1(reduct, result, reduct, vl);
  // reduct = vredsum_vs_i32m8_i32m1(reduct, lhs_value, reduct, vl);
  // q31_t res_scalar = vmv_x_s_i32m1_i32(reduct);
  printf("lhs_value=%d\n", lhs_value);
  q31_t res_scalar = 42;
  printf("res_scalar=%d\n", res_scalar);
  // *dst = (q7_t)res_scalar;
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
