# 0.6.5 测试迁移审计
基线：`34d5138aa771f707ebd300bed34aa2ccad32c478`。保留原断言除非下表明确列为由更强契约替代。历史技术论文不追改测试文件名；活动技术注册表的测试路径同步更新。
## 文件迁移
| 旧位置 | 当前位置 |
|---|---|
| `tests/test_v030_frontend_errors_resampling.py` | `tests/contracts/test_frontend_errors.py` |
| `tests/test_v040_reporting_config.py` | `tests/contracts/test_configuration.py` |
| `tests/test_econhdfe_architecture.py` | `tests/contracts/test_namespace.py` |
| `tests/test_v049_factorvars.py` | `tests/contracts/test_factor_variables.py` |
| `tests/test_v0410_fv_absorb.py` | `tests/contracts/test_absorb_syntax.py` |
| `tests/test_v0481_error_report_contract.py` | `tests/contracts/test_support_privacy.py` |
| `tests/test_support_reports.py` | `tests/contracts/test_support_reports.py` |
| `tests/test_core.py` | `tests/behavior/test_linear_models.py` |
| `tests/test_v03.py` | `tests/behavior/test_inference_semantics.py` |
| `tests/test_v04.py` | `tests/behavior/test_weights_and_effects.py` |
| `tests/test_v05.py` | `tests/behavior/test_group_individual.py` |
| `tests/test_v06.py` | `tests/behavior/test_projection_backends.py` |
| `tests/test_v07.py` | `tests/behavior/test_absorb_semantics.py` |
| `tests/test_v072_collinearity.py` | `tests/behavior/test_collinearity.py` |
| `tests/test_v073_structural_collinearity.py` | `tests/behavior/test_structural_omissions.py` |
| `tests/test_v074_dependency_omission.py` | `tests/behavior/test_dependency_omissions.py` |
| `tests/test_v075_omission_control.py` | `tests/behavior/test_user_omissions.py` |
| `tests/test_v080_execution.py` | `tests/behavior/test_execution_semantics.py` |
| `tests/test_v090_shared_iterative.py` | `tests/behavior/test_weight_updates.py` |
| `tests/test_v041_repeated_specs.py` | `tests/behavior/test_sessions.py` |
| `tests/test_v045_ppml_execution.py` | `tests/behavior/test_ppml_execution.py` |
| `tests/test_v047_reghdfe_r2.py` | `tests/behavior/test_fit_degrees_of_freedom.py` |
| `tests/test_v048_secondary_statistics.py` | `tests/behavior/test_fit_statistics.py` |
| `tests/test_cluster_inference_v046.py` | `tests/behavior/test_cluster_inference.py` |
| `tests/test_data_layer.py` | `tests/behavior/test_data_layer.py` |
| `tests/test_effects_adapters.py` | `tests/behavior/test_effects_adapters.py` |
| `tests/test_effects_diagnostics.py` | `tests/behavior/test_effects_diagnostics.py` |
| `tests/test_effects_recovery.py` | `tests/behavior/test_effects_recovery.py` |
| `tests/test_execution_planner.py` | `tests/behavior/test_execution_planner.py` |
| `tests/test_iv_ppml_core.py` | `tests/behavior/test_ivppml.py` |
| `tests/test_iv_ppml_spj.py` | `tests/behavior/test_ivppml_spj.py` |
| `tests/test_ppml_core.py` | `tests/behavior/test_ppml.py` |
| `tests/test_ppml_heterogeneous.py` | `tests/behavior/test_ppml_structured.py` |
| `tests/test_heterogeneous_model_integration.py` | `tests/behavior/test_structured_models.py` |
| `tests/test_planner_calibration_report.py` | `tests/behavior/test_planner_feedback.py` |
| `tests/test_block_design_ops.py` | `tests/numerics/test_block_algebra.py` |
| `tests/test_block_projection.py` | `tests/numerics/test_block_projection.py` |
| `tests/test_design_execution_plan.py` | `tests/numerics/test_design_certificates.py` |
| `tests/test_fe_canonicalization.py` | `tests/numerics/test_fe_canonicalization.py` |
| `tests/test_hdfe_complex_fe_corpus.py` | `tests/numerics/test_complex_fe_corpus.py` |
| `tests/test_hdfe_exact_multiway.py` | `tests/numerics/test_exact_multiway.py` |
| `tests/test_hdfe_multiway_solver_opt.py` | `tests/numerics/test_residual_core.py` |
| `tests/test_iv_shared_architecture.py` | `tests/numerics/test_iv_primitives.py` |
| `tests/test_math_resource_validation.py` | `tests/numerics/test_resource_boundaries.py` |
| `tests/test_mathematical_review.py` | `tests/numerics/test_exact_oracles.py` |
| `tests/test_native_rank_budget.py` | `tests/numerics/test_native_rank_budget.py` |
| `tests/test_nonlinear_release_validation.py` | `tests/numerics/test_nonlinear_oracles.py` |
| `tests/test_ppml_simplex_internal.py` | `tests/numerics/test_separation_corpus.py` |
| `tests/test_ppml_standardize.py` | `tests/numerics/test_standardization.py` |
| `tests/test_release_audit.py` | `tests/numerics/test_estimator_oracles.py` |
| `tests/test_release_hardening.py` | `tests/numerics/test_inference_boundaries.py` |
| `tests/test_stable_linalg_workspace.py` | `tests/numerics/test_stable_linalg.py` |
| `tests/test_structural_hardening.py` | `tests/numerics/test_block_boundaries.py` |
| `tests/test_symbolic_design_plan.py` | `tests/numerics/test_symbolic_design.py` |
| `tests/test_weighted_colsum_fusion.py` | `tests/numerics/test_weighted_reductions.py` |
| `tests/test_architecture_visualization.py` | `tests/tooling/test_architecture_map.py` |
| `tests/test_economic_module_map.py` | `tests/tooling/test_module_registry.py` |
| `tests/test_execution_acceptance.py` | `tests/tooling/test_execution_acceptance.py` |
| `tests/test_followup_release_helpers.py` | `tests/tooling/test_cli_helpers.py` |
| `tests/test_release_engineering.py` | `tests/tooling/test_release_artifacts.py` |
| `tests/test_repository_layout.py` | `tests/tooling/test_repository_layout.py` |
| `tests/test_technical_innovation_registry.py` | `tests/tooling/test_technical_registry.py` |
| `tests/test_v042_release_maintenance.py` | `tests/tooling/test_maintenance.py` |
| `tests/test_v043_skill_compatibility.py` | `tests/tooling/test_compatibility.py` |
| `tests/test_v045_real_world_benchmark_registry.py` | `tests/tooling/test_benchmark_registry.py` |
| `tests/test_v048_stats_skill_benchmark.py` | `tests/tooling/test_validation_policy.py` |
| `tests/test_versioning_policy.py` | `tests/tooling/test_versioning_policy.py` |

## 移除或替换的冗余函数
| 原测试 | 当前覆盖与理由 |
|---|---|
| `tests/test_core.py::test_ols_matches_dummy_regression` | numerics/test_estimator_oracles.py: explicit dummy WLS/IV, including beta and VCE rather than Monte Carlo proximity |
| `tests/test_core.py::test_iv_recovers_structural_coefficient` | numerics/test_estimator_oracles.py: explicit dummy WLS/IV, including beta and VCE rather than Monte Carlo proximity |
| `tests/test_core.py::test_iv_matches_explicit_dummy_2sls` | numerics/test_estimator_oracles.py: explicit dummy WLS/IV, including beta and VCE rather than Monte Carlo proximity |
| `tests/test_core.py::test_oneway_cluster_vcov_matches_full_dummy_statsmodels` | numerics/test_estimator_oracles.py: explicit dummy WLS/IV, including beta and VCE rather than Monte Carlo proximity |
| `tests/test_core.py::test_weighted_ols_matches_weighted_dummy_regression` | numerics/test_estimator_oracles.py: explicit dummy WLS/IV, including beta and VCE rather than Monte Carlo proximity |
| `tests/test_core.py::test_singletons_are_iterated` | contracts/test_public_workflows.py and behavior/test_sample_semantics.py: exact sample/report contracts |
| `tests/test_core.py::test_iv_diagnostics_are_exposed` | contracts/test_public_workflows.py and behavior/test_sample_semantics.py: exact sample/report contracts |
| `tests/test_econhdfe_architecture.py::test_primary_and_legacy_estimator_names_are_consistent` | contracts/test_public_workflows.py: all current public aliases and all four estimator families |
| `tests/test_econhdfe_architecture.py::test_new_namespace_smoke_ols_and_iv` | contracts/test_public_workflows.py: all current public aliases and all four estimator families |
| `tests/test_econhdfe_architecture.py::test_dependency_boundaries_are_one_way` | tooling/test_dependency_contract.py: actual AST imports, absolute and relative |
| `tests/test_econhdfe_architecture.py::test_generic_iv_uses_public_compute_primitives_only` | tooling/test_dependency_contract.py: actual AST imports, absolute and relative |
| `tests/test_econhdfe_architecture.py::test_ivppml_dependency_direction_is_model_level_only` | tooling/test_dependency_contract.py: actual AST imports, absolute and relative |
| `tests/test_iv_shared_architecture.py::test_iv_dependency_direction_is_model_agnostic` | tooling/test_dependency_contract.py: actual AST import graph |
| `tests/test_v030_frontend_errors_resampling.py::test_string_fixed_effect_is_valid_input_role` | contracts/test_public_workflows.py: string FE named/array/reusable parity |
| `tests/test_v030_frontend_errors_resampling.py::test_resampling_layer_has_no_model_or_hdfe_dependency` | tooling/test_dependency_contract.py: actual AST import graph |
| `tests/test_v040_reporting_config.py::test_ols_publication_output_has_top5_core_fields_and_hides_diagnostics` | contracts/test_public_workflows.py: parameterized actual result/report protocol and estimator-specific fields |
| `tests/test_v040_reporting_config.py::test_iv_publication_output_keeps_reportable_first_stage_not_fitted_matrix` | contracts/test_public_workflows.py: parameterized actual result/report protocol and estimator-specific fields |
| `tests/test_v040_reporting_config.py::test_ppml_and_ivppml_publication_outputs_are_model_specific` | contracts/test_public_workflows.py: parameterized actual result/report protocol and estimator-specific fields |
| `tests/test_v049_factorvars.py::test_fv_is_available_through_legacy_compat_namespace` | contracts/test_public_workflows.py: every exported symbol is the same legacy object |
| `tests/test_ppml_core.py::test_offset_and_exposure_equivalent` | contracts/test_public_workflows.py: reusable/direct, offset/exposure, current snapshot rather than a pinned historical default |
| `tests/test_ppml_core.py::test_reusable_model_reuses_compiled_plan` | contracts/test_public_workflows.py: reusable/direct, offset/exposure, current snapshot rather than a pinned historical default |
| `tests/test_ppml_core.py::test_public_default_maxiter_matches_ppmlhdfe_command` | contracts/test_public_workflows.py: reusable/direct, offset/exposure, current snapshot rather than a pinned historical default |
| `tests/test_iv_ppml_core.py::test_ivppml_offset_and_exposure_are_equivalent` | contracts/test_public_workflows.py: current nonlinear interface workflow |
| `tests/test_iv_ppml_core.py::test_ivppml_dataframe_model_reuses_plan_and_cluster_vce` | contracts/test_public_workflows.py: current nonlinear interface workflow |
| `tests/test_v043_skill_compatibility.py::test_public_contract_snapshot_has_required_surfaces` | tooling/test_current_release_contract.py: live current-version snapshot plus one aggregate version/compatibility/Skill gate |
| `tests/test_v043_skill_compatibility.py::test_public_contract_gate_accepts_current_release` | tooling/test_current_release_contract.py: live current-version snapshot plus one aggregate version/compatibility/Skill gate |
| `tests/test_v043_skill_compatibility.py::test_canonical_skill_structure_validates` | tooling/test_current_release_contract.py: live current-version snapshot plus one aggregate version/compatibility/Skill gate |
| `tests/test_v042_release_maintenance.py::test_release_maintenance_manifest_is_machine_checkable` | tooling/test_current_release_contract.py: aggregate actual gate |
| `tests/test_v041_repeated_specs.py::test_version_control_check_is_machine_executable` | tooling/test_current_release_contract.py: aggregate actual gate |
| `tests/test_release_audit.py::test_regression_lsmr_condition_stop_is_not_success` | numerics/test_inference_boundaries.py: every LSMR stop code and failure interpretation retained |
| `tests/test_v047_reghdfe_r2.py::test_repeated_ols_session_uses_same_fit_dof_statistics_as_direct` | behavior/test_sessions.py: all fit statistics checked on every multi-outcome/control/FE specification |
| `tests/test_v048_secondary_statistics.py::test_repeated_ols_secondary_stats_match_direct` | behavior/test_sessions.py: all fit statistics checked on OLS and IV session results |
| `tests/test_v048_secondary_statistics.py::test_repeated_iv_secondary_stats_match_direct` | behavior/test_sessions.py: all fit statistics checked on OLS and IV session results |
| `tests/test_v047_reghdfe_r2.py::test_fweight_r2_matches_explicit_row_duplication` | behavior/test_weights_and_effects.py: frequency replication checks beta, VCE, DoF and all fit scalars together |
| `tests/test_v048_secondary_statistics.py::test_fweight_secondary_fit_statistics_match_explicit_row_duplication` | behavior/test_weights_and_effects.py: three-VCE frequency replication checks all fit scalars |
| `tests/test_v0481_error_report_contract.py::test_release_verifier_checks_error_template_identity` | tooling/test_release_artifacts.py and actual bundle verification; template byte identity/privacy tests remain |
| `tests/test_v0481_error_report_contract.py::test_release_verifier_checks_planner_template_identity` | tooling/test_release_artifacts.py and actual bundle verification; template byte identity/privacy tests remain |
| `tests/test_repository_layout.py::test_hdfe_technical_chain_is_canonical` | tooling/test_current_release_contract.py, test_technical_registry.py and test_architecture_map.py: executable registry/path/content verification |
| `tests/test_repository_layout.py::test_release_governance_uses_nested_canonical_manifest` | tooling/test_current_release_contract.py, test_technical_registry.py and test_architecture_map.py: executable registry/path/content verification |
| `tests/test_repository_layout.py::test_generated_architecture_map_is_in_development_docs` | tooling/test_current_release_contract.py, test_technical_registry.py and test_architecture_map.py: executable registry/path/content verification |

## 缩减的参数组合
这些缩减不声称保留所有高阶因素组合；保留对应输入域和独立异常守卫。

### `tests/test_stable_linalg_workspace.py::test_dense_svd_oracle_layout_rank_and_input_ownership`
Pairwise shape/layout/deficiency coverage; extreme units, last-block nonfinite, empty, read-only and minimum-norm guards remain separate

原组合：
```python
@pytest.mark.parametrize("shape", [(17, 5), (9, 14), (123, 6)])
@pytest.mark.parametrize("layout", ["C", "F", "sliced", "readonly"])
@pytest.mark.parametrize("deficient", [False, True])
```

新组合：
```python
@pytest.mark.parametrize('shape,layout,deficient', [
    ((17, 5), "C", False), ((17, 5), "F", True), ((17, 5), "sliced", False), ((17, 5), "readonly", True),
    ((9, 14), "C", True), ((9, 14), "F", False), ((9, 14), "sliced", True), ((9, 14), "readonly", False),
    ((123, 6), "C", False), ((123, 6), "F", True), ((123, 6), "sliced", True), ((123, 6), "readonly", False),
])
```

### `tests/test_weighted_colsum_fusion.py::test_layout_empty_and_input_preservation`
All layout/shape pairs and both weighted paths for every shape/layout; dtype/overflow/dynamic-weight guards unchanged

原组合：
```python
@pytest.mark.parametrize('order', ['C', 'F', 'slice', 'reverse'])
@pytest.mark.parametrize('shape', [(37, 5), (0, 3), (6, 0), (1, 7)])
@pytest.mark.parametrize('weighted', [False, True])
```

新组合：
```python
@pytest.mark.parametrize('order,shape,weighted', [
    (order, shape, (i+j)%2 == 0)
    for i, order in enumerate(("C", "F", "slice", "reverse"))
    for j, shape in enumerate(((37,5), (0,3), (6,0), (1,7)))
])
```

### `tests/test_release_hardening.py::test_iv_role_scaling_preserves_beta_vcov_and_first_stage`
Every estimator/VCE pair retained; both scale directions appear per estimator and VCE; LIML outcome-scaling checks unchanged

原组合：
```python
@pytest.mark.parametrize('estimator', ['2sls','liml','kclass','gmm2s'])
@pytest.mark.parametrize('scale', [1e-10,1e10])
@pytest.mark.parametrize('vce', ['iid','robust','cluster'])
```

新组合：
```python
@pytest.mark.parametrize('estimator,scale,vce', [
    (model, 1e-10 if (i+j)%2 else 1e10, vce)
    for i, model in enumerate(("2sls", "liml", "kclass", "gmm2s"))
    for j, vce in enumerate(("iid", "robust", "cluster"))
])
```

### `tests/test_release_audit.py::test_exact_categorical_rank_against_dense_svd`
One through four FE dimensions; stronger exhaustive Fraction and random multiway exact oracle retained

原组合：
```python
@pytest.mark.parametrize('seed',range(20))
```

新组合：
```python
@pytest.mark.parametrize('seed', [0, 1, 2, 3])
```

### `tests/test_native_rank_budget.py::test_native_permuted_labels_and_bad_prime_match_exact_oracle`
Retain 3/4-way bad-prime/relabel/order oracle; resource failure branches unchanged

原组合：
```python
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("width", [3, 4])
```

新组合：
```python
@pytest.mark.parametrize('seed,width', [(0,3), (1,4), (2,3), (3,4)])
```

### `tests/test_structural_hardening.py::test_partitioned_wls_randomized_full_rank_parity`
Reduce repeated seeds; shared border, disjoint, deficiency, fallback, stability and memory cases retained

原组合：
```python
@pytest.mark.parametrize("seed", range(12))
```

新组合：
```python
@pytest.mark.parametrize('seed', [0, 1, 2, 3])
```

### `tests/test_nonlinear_release_validation.py::test_ivppml_units_preserve_covariance`
Pairwise path/VCE/role coverage; all original scale magnitudes retained

原组合：
```python
@pytest.mark.parametrize('path', ['dense', 'block'])
@pytest.mark.parametrize('kind', ['robust', 'cluster'])
@pytest.mark.parametrize('role,scale', [('exog', 4e7), ('endog', 4e7), ('instruments', 1e-7)])
```

新组合：
```python
@pytest.mark.parametrize('path,kind,role,scale', [('dense','robust','exog',4e7), ('dense','cluster','endog',4e7), ('dense','robust','instruments',1e-7), ('block','cluster','exog',4e7), ('block','robust','endog',4e7), ('block','cluster','instruments',1e-7)])
```

### `tests/test_math_resource_validation.py::test_equilibrated_qr_matches_explicit_scaled_reference`
Named tall/underdetermined/square/empty/chunk-boundary cases; independent rank-deficient chunks remain

原组合：
```python
@pytest.mark.parametrize('shape',[(80,4),(4,8),(2,2),(0,3),(6,0)])
@pytest.mark.parametrize('chunk',[1,17,32768])
```

新组合：
```python
@pytest.mark.parametrize('shape,chunk', [((80,4),1), ((80,4),32768), ((4,8),17), ((2,2),1), ((0,3),17), ((6,0),32768)])
```

### `tests/test_math_resource_validation.py::test_ppml_block_covariance_preserves_units`
Remove identity transform (scale=1), which reran identical fits and asserted self-equality

原组合：
```python
@pytest.mark.parametrize('kind',['model','robust','cluster'])
@pytest.mark.parametrize('scale',[1.,4e7])
```

新组合：
```python
@pytest.mark.parametrize('kind,scale', [('model',4e7), ('robust',4e7), ('cluster',4e7)])
```

### `tests/test_math_resource_validation.py::test_block_iv_rank_and_bread_use_the_same_equilibrated_coordinates`
Remove identity transform, retain small/large coordinate changes

原组合：
```python
@pytest.mark.parametrize('scale',[1e-8,1.,4e7])
```

新组合：
```python
@pytest.mark.parametrize('scale', [1e-8,4e7])
```

## 额外迁移检查
逐行/逐分支比较发现多向 cluster 的 session 来源列分支和可选 FE 诊断分支原本会丢失；已分别纳入现有多 outcome 工作流与停止码测试。没有仅用覆盖率净增长抵消丢失的分支。
