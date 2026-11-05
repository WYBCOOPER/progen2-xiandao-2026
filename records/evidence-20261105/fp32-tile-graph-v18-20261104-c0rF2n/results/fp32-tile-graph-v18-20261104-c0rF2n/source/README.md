# v18: 256-position tiles in the v17 fused attention kernel

Parent v17. The fused B>=4 path uses 256-position tiles and 8 warps, versus
128 positions and 4 warps in v17. The Triton kernel body is byte-for-byte
unchanged. Larger tiles reduce online-softmax loop iterations; more warps
distribute the larger tile. Occupancy/resource cost can offset the benefit,
so speed is unknown until DCU measurement. B<4 retains the v17/v15 path.

Keep FP32, explicit K/V mutation schema, full capacity 1024, compiled HIP
graph replay, per-request cache reset, greedy rule and official tools.
Tile width changes reduction grouping, so full public exactness is required.
Preflight replays at positions 0,255,511,1022, checking tile boundaries and
the final valid slot. Formal compile/miss/capture counters must be zero.

Run from progen2_contest:
bash candidates/fp32_tile_graph_v18/run_test.sh

Download RESULT_ARCHIVE on success or failure. v17 remains separate.
Local kernel emulation cannot establish actual HIP performance or full-
model correctness. Graph runtime code is unchanged from DCU-passed v17.
