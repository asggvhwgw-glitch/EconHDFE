import EconHDFE.LeastSquaresExistence

/-!
# Finite weighted Frisch--Waugh--Lovell theory

This connects minimizers on the original FE-plus-regressor space with the
within problem. It is a mathematical equivalence, not a numerical solver
specification. Redundant FE coefficient coordinates are never assumed unique.
-/
noncomputable section
namespace EconHDFE
open scoped Classical
variable {I E : Type*} [Fintype I] [AddCommGroup E] [Module ℝ E]

/-- A full WLS optimum induces a WLS optimum in the transformed problem. -/
theorem full_wls_implies_within (w : I → ℝ) (hw : ∀ i, 0 < w i)
    (S : Submodule ℝ (I → ℝ)) (X : E →ₗ[ℝ] (I → ℝ))
    (y : I → ℝ) (beta : E) (d : I → ℝ) (hd : d ∈ S)
    (hf : IsWLSFit w (S ⊔ LinearMap.range X) y (X beta + d)) :
    IsWLSFit w (LinearMap.range ((wResidualMap w hw S).comp X))
      (wResidualMap w hw S y) (((wResidualMap w hw S).comp X) beta) := by
  let M := wResidualMap w hw S
  have hr := wlsFit_isResidual hw hf
  have hprof : M (y - X beta) = y - (X beta + d) := by
    apply residual_unique hw (wResidual_spec w hw S (y - X beta))
    constructor
    · convert hd using 1 <;> abel
    · intro z hz
      exact hr.2 z ((show S ≤ S ⊔ LinearMap.range X from le_sup_left) hz)
  refine ⟨⟨beta, rfl⟩, ?_⟩
  rintro g ⟨gamma, rfl⟩
  let r := M (y - X gamma)
  let s := y - X gamma - r
  have hs : s ∈ S := (wResidual_spec w hw S (y - X gamma)).1
  have hc : X gamma + s ∈ S ⊔ LinearMap.range X :=
    (S ⊔ LinearMap.range X).add_mem
      ((show LinearMap.range X ≤ S ⊔ LinearMap.range X from le_sup_right) ⟨gamma, rfl⟩)
      ((show S ≤ S ⊔ LinearMap.range X from le_sup_left) hs)
  have hglobal := hf.2 (X gamma + s) hc
  have hrg : y - (X gamma + s) = r := by dsimp [s]; abel
  have h0 : wInner w (M (y - X beta)) (M (y - X beta)) ≤ wInner w r r := by
    rw [hprof]
    rwa [hrg] at hglobal
  simpa only [r, map_sub, LinearMap.comp_apply] using h0

/-- A within optimum lifts to an attainable full optimum. -/
theorem within_wls_implies_full (w : I → ℝ) (hw : ∀ i, 0 < w i)
    (S : Submodule ℝ (I → ℝ)) (X : E →ₗ[ℝ] (I → ℝ))
    (y : I → ℝ) (beta : E)
    (hf : IsWLSFit w (LinearMap.range ((wResidualMap w hw S).comp X))
      (wResidualMap w hw S y) (((wResidualMap w hw S).comp X) beta)) :
    IsWLSFit w (S ⊔ LinearMap.range X) y
      (y - wResidualMap w hw S (y - X beta)) := by
  let M := wResidualMap w hw S
  let r := M (y - X beta)
  have hS : IsResidual w S (y - X beta) r := wResidual_spec w hw S (y - X beta)
  have hW : IsResidual w (LinearMap.range (M.comp X)) (M y) r := by
    simpa only [r, map_sub, LinearMap.comp_apply] using wlsFit_isResidual hw hf
  apply residual_isWLSFit (fun i => (hw i).le)
  constructor
  · have hm := (S ⊔ LinearMap.range X).add_mem
      ((show S ≤ S ⊔ LinearMap.range X from le_sup_left) hS.1)
      ((show LinearMap.range X ≤ S ⊔ LinearMap.range X from le_sup_right) ⟨beta, rfl⟩)
    convert hm using 1 <;> abel
  · intro z hz
    obtain ⟨s, hs, x, hx, hsum⟩ := Submodule.mem_sup.mp hz
    rw [← hsum, wInner_add_right, hS.2 s hs, zero_add]
    obtain ⟨gamma, rfl⟩ := hx
    have hrem := hS.2 (X gamma - M (X gamma)) (wResidual_spec w hw S (X gamma)).1
    have hzW := hW.2 (M (X gamma)) ⟨gamma, rfl⟩
    rw [wInner_sub_right, hzW, sub_zero] at hrem
    exact hrem

/-- Equal FE spaces preserve identified non-FE coefficients of full WLS optima. -/
theorem full_wls_identified_fe_invariance (w : I → ℝ) (hw : ∀ i, 0 < w i)
    {S T : Submodule ℝ (I → ℝ)} (hST : S = T)
    (X : E →ₗ[ℝ] (I → ℝ)) (y : I → ℝ) (beta gamma : E)
    (d e : I → ℝ) (hd : d ∈ S) (he : e ∈ T)
    (hb : IsWLSFit w (S ⊔ LinearMap.range X) y (X beta + d))
    (hg : IsWLSFit w (T ⊔ LinearMap.range X) y (X gamma + e))
    (identified : Function.Injective ((wResidualMap w hw S).comp X)) : beta = gamma := by
  subst T
  have hb' := full_wls_implies_within w hw S X y beta d hd hb
  have hg' := full_wls_implies_within w hw S X y gamma e he hg
  exact identified (wls_fitted_invariance hw rfl hb' hg')

/-- An estimable coefficient functional is constant on each exact fitted-value fiber. -/
theorem estimable_coefficient_fiber (X : E →ₗ[ℝ] (I → ℝ)) (ell : E →ₗ[ℝ] ℝ)
    (estimable : LinearMap.ker X ≤ LinearMap.ker ell) {beta gamma : E}
    (same : X beta = X gamma) : ell beta = ell gamma := by
  have hx : beta - gamma ∈ LinearMap.ker X := by
    change X (beta - gamma) = 0
    rw [map_sub, same, sub_self]
  have he := estimable hx
  change ell (beta - gamma) = 0 at he
  rw [map_sub] at he
  exact sub_eq_zero.mp he

end EconHDFE
