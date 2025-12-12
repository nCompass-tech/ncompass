//! Event linking algorithms for NVTX-kernel correlation

pub mod adapters;
pub mod algorithms;
pub mod nvtx_linker;

pub use adapters::{EventAdapter, NsysEventAdapter};
pub use algorithms::{
    aggregate_kernel_times, build_correlation_map, find_kernels_for_annotation,
    find_overlapping_intervals,
};
pub use nvtx_linker::link_nvtx_to_kernels;

