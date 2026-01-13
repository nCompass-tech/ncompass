#!/usr/bin/env python3
# Copyright 2025 nCompass Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
nCompass Profiling - Nsight Compute (ncu) integration.

Profiling config
"""


from dataclasses import dataclass

@dataclass(frozen=True)
class ProfilingConfig:
    """Profiling configuration."""
    ncu_metrics: tuple[str, ...] = (
        "gpu__time_duration.sum",
        "sm__cycles_elapsed.avg.per_second",
        "smsp__cycles_elapsed.avg.per_second",
        "dram__cycles_elapsed.avg.per_second",
        "lts__cycles_elapsed.avg.per_second",
        "l1tex__cycles_elapsed.avg.per_second",
        "dram__bytes.sum.peak_sustained",
        "lts__lts2xbar_cycles_active.sum.peak_sustained",
        "lts__lts2xbar_cycles_active.avg.pct_of_peak_sustained_elapsed",
        "l1tex__lsu_writeback_active_mem_lgds.sum.peak_sustained",
        "l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_elapsed",
        "sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained",
        "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed",
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained",
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed",
        "sm__sass_thread_inst_executed_op_hfma_pred_on.sum.peak_sustained",
        "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_elapsed",
        "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_elapsed",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
    )

config = ProfilingConfig()
