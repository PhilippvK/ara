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

// Define Matrix dimensions:
// o = i ° f, with i=[MxN], f=[FxF], o=[MxN]
// The filter is a square matrix, and F is odd

// Matrices defined in data.S
extern int64_t i[] __attribute__((
    aligned(4 * NR_LANES))); // [ (M+floor(F/2)) * (N+floor(F/2)) ]
extern int64_t f[] __attribute__((aligned(4 * NR_LANES)));        // [ F*F ]
extern int64_t o[] __attribute__((aligned(4 * NR_LANES)));        // [ M*N ]
extern int64_t golden_o[] __attribute__((aligned(4 * NR_LANES))); // [ M*N ]
// M, N, F defined in data.S
extern int64_t M;
extern int64_t N;
extern int64_t F;

// Verify the matrices
int verify_matrix(int64_t *matrix, int64_t *golden_matrix, int64_t R,
                  int64_t C) {
  for (int r = 0; r < R; ++r)
    for (int c = 0; c < C; ++c)
      if (matrix[c + C * r] != golden_matrix[c + C * r]) {
        printf("Error: o[%d][%d] = %ld, instead of %ld\n", r, c,
               matrix[c + C * r], golden_matrix[c + C * r]);
        return 1;
      }
  return 0;
}

void print_matrix(int64_t const *matrix, uint64_t num_rows,
                  uint64_t num_columns) {
  printf("0x%8X\n", (uint64_t)matrix);
  for (uint64_t i = 0; i < num_rows; ++i) {
    for (uint64_t j = 0; j < num_columns; ++j) {
      printf("%10d ", matrix[i * num_columns + j]);
    }
    printf("\n");
  }
}

#define N 640
// #define N 1
#define M 128
// #define M 1

int main() {
  printf("\n");
  printf("=============\n");
  printf("=  .......  =\n");
  printf("=============\n");
  printf("\n");
  printf("\n");

  // Call the main kernel, and measure cycles
  start_timer();
  const muriscv_nn_activation activation = {-128,127};
  const muriscv_nn_fc_params fc_params = {-89, 0, -128, activation};
  muriscv_nn_per_tensor_quant_params quant_params = {1638001653, -8};
  muriscv_nn_dims input_dims = {1,1,1,N};
  muriscv_nn_dims filter_dims = {N,1,1,M};
  // muriscv_nn_dims bias_dims = {1,1,1,M};
  muriscv_nn_dims output_dims = {1,1,1,M};
  static q7_t input[N];
  static q7_t kernel[N*M];
  static int32_t bias[M];
  static q7_t output[M];
  q7_t *input_ptr = &input[0];
  q7_t *kernel_ptr = &kernel[0];
  int32_t *bias_ptr = &bias[0];
  q7_t *output_ptr = &output[0];

  int32_t batch_cnt = input_dims.n;

  while (batch_cnt) {
    muriscv_nn_vec_mat_mult_t_s8(input_ptr,
                                 kernel_ptr,
                                 bias_ptr,
                                 output_ptr,
                                 fc_params.input_offset,
                                 0,
                                 fc_params.output_offset,
                                 quant_params.multiplier,
                                 quant_params.shift,
                                 filter_dims.n, /* col_dim or accum_depth */
                                 output_dims.c, /* row_dim or output_depth */
                                 fc_params.activation.min,
                                 fc_params.activation.max,
                                 1L);
    input_ptr += filter_dims.n;
    output_ptr += output_dims.c;
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
