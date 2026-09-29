import EconHDFE.LeastSquaresExistence
import EconHDFE.PartitionRefinement

/-!
# Structural reductions and within-coefficient semantics

Equal FE spaces define equal mathematical within problems. Rank-deficient
coefficient equality is asserted only for the same selector on that problem,
not for an arbitrary minimum-norm choice on differently parameterized full models.
-/
noncomputable section
namespace EconHDFE
open scoped Classical
variable {I V : Type*} [Fintype I]

def withinDesign (w : I → ℝ) (hw : ∀ i, 0 < w i)
    (S : Submodule ℝ (I → ℝ)) (X : Matrix I V ℝ) : Matrix I V ℝ :=
  fun i j => wResidual w hw S (fun k => X k j) i

theorem within_design_invariance (w : I → ℝ) (hw : ∀ i, 0 < w i)
    {S T : Submodule ℝ (I → ℝ)} (hST : S = T) (X : Matrix I V ℝ) :
    withinDesign w hw S X = withinDesign w hw T X := by
  ext i j
  exact congrFun (wResidual_eq_of_space_eq w hw hST (fun k => X k j)) i

/-- This selector receives only the common transformed problem, not full FE coordinates. -/
theorem within_selection_invariance {B : Type*}
    (choose : Matrix I V ℝ → (I → ℝ) → B)
    (w : I → ℝ) (hw : ∀ i, 0 < w i)
    {S T : Submodule ℝ (I → ℝ)} (hST : S = T) (X : Matrix I V ℝ) (y : I → ℝ) :
    choose (withinDesign w hw S X) (wResidual w hw S y) =
      choose (withinDesign w hw T X) (wResidual w hw T y) := by
  rw [within_design_invariance w hw hST X, wResidual_eq_of_space_eq w hw hST y]

/-- Identified coordinates are unique once both fits solve the same within WLS problem. -/
theorem within_identified_coefficients [Fintype V]
    (w : I → ℝ) (hw : ∀ i, 0 < w i)
    {S T : Submodule ℝ (I → ℝ)} (hST : S = T) (X : Matrix I V ℝ) (y : I → ℝ)
    (beta gamma : V → ℝ)
    (identified : Function.Injective (withinDesign w hw S X).mulVecLin)
    (hb : IsWLSFit w (LinearMap.range (withinDesign w hw S X).mulVecLin)
      (wResidual w hw S y) ((withinDesign w hw S X).mulVecLin beta))
    (hg : IsWLSFit w (LinearMap.range (withinDesign w hw T X).mulVecLin)
      (wResidual w hw T y) ((withinDesign w hw T X).mulVecLin gamma)) : beta = gamma := by
  rw [← within_design_invariance w hw hST X, ← wResidual_eq_of_space_eq w hw hST y] at hg
  exact identified (wls_fitted_invariance hw rfl hb hg)

/-- A block contained in the absorbed space has no within direction. -/
theorem absorbed_block_within_zero (w : I → ℝ) (hw : ∀ i, 0 < w i)
    {S : Submodule ℝ (I → ℝ)} (X : Matrix I V ℝ)
    (absorbed : ∀ j, (fun i => X i j) ∈ S) : withinDesign w hw S X = 0 := by
  ext i j
  exact congrFun (wResidual_mem_zero w hw (absorbed j)) i

/-- The common fitted space also identifies every well-defined linear function of fits. -/
theorem estimable_fit_function_invariance {B : Type*} [AddCommGroup B] [Module ℝ B]
    (w : I → ℝ) (hw : ∀ i, 0 < w i)
    {S T : Submodule ℝ (I → ℝ)} (hST : S = T)
    {y f g : I → ℝ} (hf : IsWLSFit w S y f) (hg : IsWLSFit w T y g)
    (ell : (I → ℝ) →ₗ[ℝ] B) : ell f = ell g :=
  congrArg ell (wls_fitted_invariance hw hST hf hg)

end EconHDFE
