import EconHDFE.ActivePartitionRefinement
import EconHDFE.WithinInvariance
import EconHDFE.PPMLInvariance

/-!
# Mathematical structural-reduction theorem

The constructors are the paper's algebraic reductions, not states of the Python
planner. They carry refinement maps, active-level coverage and retained witnesses,
never an assumed equality of the spaces to be proved equal.
-/
noncomputable section
namespace EconHDFE
open scoped Classical
universe u

inductive StructuralStep (I : Type u) :
    Submodule ℝ (I → ℝ) → Submodule ℝ (I → ℝ) → Prop where
  | canonicalize {A B : Type u} (P : I → A) (Q : I → B)
      (h : Refines Q P) (R : Submodule ℝ (I → ℝ)) :
      StructuralStep I ((R ⊔ partitionSpace P) ⊔ partitionSpace Q) (R ⊔ partitionSpace Q)
  | active {A B : Type u} (P : I → A) (Q : I → B) (f : B → A)
      (h : ∀ i, P i = f (Q i)) (a : Finset B) (q : B) (hq : q ∈ a)
      (R : Submodule ℝ (I → ℝ)) (coarse : indicator P (f q) ∈ R)
      (complete : ∀ i, P i = f q → Q i ∈ a) :
      StructuralStep I (R ⊔ activeIndicatorSpace Q a)
        (R ⊔ activeIndicatorSpace Q (a.erase q))
  | multiply (m : I → ℝ) {S T : Submodule ℝ (I → ℝ)} (step : StructuralStep I S T) :
      StructuralStep I (S.map (multiplier m)) (T.map (multiplier m))
  | context (R : Submodule ℝ (I → ℝ)) {S T : Submodule ℝ (I → ℝ)}
      (step : StructuralStep I S T) : StructuralStep I (R ⊔ S) (R ⊔ T)

inductive StructuralTrace (I : Type u) :
    Submodule ℝ (I → ℝ) → Submodule ℝ (I → ℝ) → Prop where
  | refl (S : Submodule ℝ (I → ℝ)) : StructuralTrace I S S
  | cons {S T U : Submodule ℝ (I → ℝ)}
      (step : StructuralStep I S T) (rest : StructuralTrace I T U) : StructuralTrace I S U

variable {I : Type u}

theorem structural_step_space_eq {S T : Submodule ℝ (I → ℝ)}
    (step : StructuralStep I S T) : S = T := by
  induction step with
  | canonicalize P Q h R => exact canonicalize_one h R
  | active P Q f h a q hq R coarse complete =>
    exact drop_active_fine P Q f h a q hq R coarse complete
  | multiply m step ih => exact common_multiplier_space_eq m ih
  | context R step ih => exact reduction_in_context R ih

/-- Main column-space theorem for a finite sequence of the allowed reductions. -/
theorem structural_trace_space_eq {S T : Submodule ℝ (I → ℝ)}
    (trace : StructuralTrace I S T) : S = T := by
  induction trace with
  | refl => rfl
  | cons step rest ih => exact (structural_step_space_eq step).trans ih

theorem structural_trace_within [Fintype I] (w : I → ℝ) (hw : ∀ i, 0 < w i)
    {S T : Submodule ℝ (I → ℝ)} (trace : StructuralTrace I S T) (y : I → ℝ) :
    wResidual w hw S y = wResidual w hw T y :=
  wResidual_eq_of_space_eq w hw (structural_trace_space_eq trace) y

theorem structural_trace_wls_fit [Fintype I] (w : I → ℝ) (hw : ∀ i, 0 < w i)
    {S T : Submodule ℝ (I → ℝ)} (trace : StructuralTrace I S T)
    {y f g : I → ℝ} (hf : IsWLSFit w S y f) (hg : IsWLSFit w T y g) : f = g :=
  wls_fitted_invariance hw (structural_trace_space_eq trace) hf hg

theorem structural_trace_ppml [Fintype I] (w : I → ℝ) (hw : ∀ i, 0 < w i)
    {S T : Submodule ℝ (I → ℝ)} (trace : StructuralTrace I S T)
    {y offset eta theta : I → ℝ} (he : IsPPMLFit w y offset S eta)
    (ht : IsPPMLFit w y offset T theta) : eta = theta :=
  finite_ppml_fitted_invariance hw (structural_trace_space_eq trace) he ht

end EconHDFE
