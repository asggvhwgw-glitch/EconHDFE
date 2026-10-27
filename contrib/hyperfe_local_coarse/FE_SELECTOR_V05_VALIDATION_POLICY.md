# 0.4 frozen validation policy

- Development: old seed 73021; sample budgets 65536 and 131072 inspected and
  retained. Default 131072 fixed before first exploratory seed 83031.
- Main: 1,000,000 observations; six structural cases; native Adaptive, frozen
  v03 and v04; 3 independent processes each; deterministic shuffled order,
  sequential execution; all timed work and failed choices retained.
- Supplemental seed: 83132; not a new data-generating mechanism.
- Full-design independent LSMR reference on weighted12, equal12, smooth3.
  Original known residual, normal and feasibility audits on all cases.
- Forward oracle <1e-8; all-column relative reference error <1e-8 where present;
  no tolerance relaxation after results. Both use float64 and default benchmark
  native/selector tol=1e-13; reference tol=1e-14.
- Same optimized original-row audit on native control and selectors. Native
  control is a direct solve_batch, not routed through a selector.
- Same input hash required across compared methods. Child environment established
  before imports; loaded BLAS libraries audited after solve. Single-thread CPUs.
- Setup, scouting, confirmation, failed exploration, solves, reconstruction and
  runtime audit included. Data generation, exact tuple-count metadata, JIT warmup
  and external reference checking excluded and disclosed.
- Workspace ledger is not a process-RSS hard cap. Sample coverage is not a rank
  or spectral-equivalence certificate. Reject/unknown/native fallback != a local
  solver victory. Native-path timing gains are not new-algorithm speedups.
- Main source hashes frozen in results/SOURCE_FREEZE.json. Any change requires
  a new labelled development phase, new freeze, and honest rerun boundaries.

Revision A used 83031 and revealed an irregular-community false acceptance. The
24 completed comparisons and 3 references are retained outside final statistics.
Final main seed is 83131; final supplementary seed is 83132.

## 补充诊断（不修改冻结策略）

主比较暴露连续权重seed83131的24步试探失败后，另测48/96步，
不得用这些结果替换主表。seed83132的代理拒绝之后，另计算完整参考，
对照原生、完整0.4和关闭前置抽样的0.4；用于检查误拒绝，
不再调整算法或数据生成器。相应任务记录在rejection_diagnostics_jobs.json。
