# tests/README

## 待办:exllamav3_ext 容器内复验(需 GPU 维护窗口)

test_exl3 的 golden hash 首跑时是 PENDING(占位)。流程:
1. 维护窗口内起 poc 容器(见 ../bench/poc_exl3_format.py 的跑法)
2. 跑 PoC 确认 ext 逐位一致后,把 decode_inner(gate_proj) 的 sha256 填入 _GOLDEN_GATE_INNER
3. 此后 golden 常态防回归

另有锚点断言(test_mcg_table_invariants)防语义漂移。
