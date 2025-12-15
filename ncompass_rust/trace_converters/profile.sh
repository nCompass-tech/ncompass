function run_profile() {
  sudo -E /home/ncuser/.cargo/bin/cargo flamegraph \
    --profile profiling \
    --bin nsys-chrome \
    -- /home/ubuntu/adi/ncompass/examples/trace_converter/.traces/nsys_h200_vllm_qwen30ba3b_TP1_quant.nsys-rep \
          -o ./tmp.json.gz \
          -t kernel,nvtx,nvtx-kernel,cuda-api,osrt,sched
}

function cleanup() {
  rm perf.*
  rm tmp.json.gz
  rm flamegraph.svg
}

run_profile
# cleanup
