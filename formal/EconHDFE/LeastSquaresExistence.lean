import EconHDFE.WeightedLeastSquares

/-!
# Existence of finite positive-weight least-squares projections

The normal-equation map into the dual of the fitted space is injective by
positive definiteness and surjective by finite dimension. This proves existence
without assuming an optimizer or verifying a numerical solver.
-/
noncomputable section
namespace EconHDFE
open scoped BigOperators Classical
variable {I : Type*} [Fintype I]

def weightedFunctional (w y : I → ℝ) (S : Submodule ℝ (I → ℝ)) :
    Module.Dual ℝ S where
  toFun z := wInner w y z.val
  map_add' z t := wInner_add_right w y z.val t.val
  map_smul' a z := wInner_smul_right w y z.val a

def weightedRiesz (w : I → ℝ) (S : Submodule ℝ (I → ℝ)) :
    S →ₗ[ℝ] Module.Dual ℝ S where
  toFun f := weightedFunctional w f.val S
  map_add' f g := by
    apply LinearMap.ext
    intro z
    exact wInner_add_left w f.val g.val z.val
  map_smul' a f := by
    apply LinearMap.ext
    intro z
    exact wInner_smul_left w f.val z.val a

theorem weightedRiesz_injective {w : I → ℝ} (hw : ∀ i, 0 < w i)
    (S : Submodule ℝ (I → ℝ)) : Function.Injective (weightedRiesz w S) := by
  intro f g h
  apply Subtype.ext
  apply sub_eq_zero.mp
  apply wInner_self_eq_zero hw
  have hz := LinearMap.congr_fun h (f - g)
  change wInner w f.val (f.val - g.val) = wInner w g.val (f.val - g.val) at hz
  rw [wInner_sub_left, hz, sub_self]

theorem weightedRiesz_surjective {w : I → ℝ} (hw : ∀ i, 0 < w i)
    (S : Submodule ℝ (I → ℝ)) : Function.Surjective (weightedRiesz w S) := by
  apply LinearMap.range_eq_top.mp
  apply Submodule.eq_top_of_finrank_eq
  have hk : LinearMap.ker (weightedRiesz w S) = ⊥ :=
    LinearMap.ker_eq_bot.mpr (weightedRiesz_injective hw S)
  have hd := LinearMap.finrank_range_add_finrank_ker (weightedRiesz w S)
  rw [hk] at hd
  simpa using hd

/-- Every finite positive diagonal-weight problem has an attainable residual. -/
theorem wls_residual_exists {w : I → ℝ} (hw : ∀ i, 0 < w i)
    (S : Submodule ℝ (I → ℝ)) (y : I → ℝ) : ∃ r, IsResidual w S y r := by
  obtain ⟨f, hf⟩ := weightedRiesz_surjective hw S (weightedFunctional w y S)
  refine ⟨y - f.val, ?_, ?_⟩
  · simpa only [sub_sub_cancel] using f.property
  · intro z hz
    have heq := LinearMap.congr_fun hf ⟨z, hz⟩
    change wInner w f.val z = wInner w y z at heq
    rw [wInner_sub_left, heq, sub_self]

theorem wls_residual_exists_unique {w : I → ℝ} (hw : ∀ i, 0 < w i)
    (S : Submodule ℝ (I → ℝ)) (y : I → ℝ) : ∃! r, IsResidual w S y r := by
  obtain ⟨r, hr⟩ := wls_residual_exists hw S y
  exact ⟨r, hr, fun s hs => residual_unique hw hs hr⟩

theorem wls_fit_exists_unique {w : I → ℝ} (hw : ∀ i, 0 < w i)
    (S : Submodule ℝ (I → ℝ)) (y : I → ℝ) : ∃! f, IsWLSFit w S y f := by
  obtain ⟨r, hr⟩ := wls_residual_exists hw S y
  have hf := residual_isWLSFit (fun i => (hw i).le) hr
  exact ⟨y - r, hf, fun g hg => wls_fitted_invariance hw rfl hg hf⟩

/-- The unique mathematical residual; no numerical algorithm is specified. -/
def wResidual (w : I → ℝ) (hw : ∀ i, 0 < w i)
    (S : Submodule ℝ (I → ℝ)) (y : I → ℝ) : I → ℝ :=
  Classical.choose (wls_residual_exists hw S y)

theorem wResidual_spec (w : I → ℝ) (hw : ∀ i, 0 < w i)
    (S : Submodule ℝ (I → ℝ)) (y : I → ℝ) :
    IsResidual w S y (wResidual w hw S y) :=
  Classical.choose_spec (wls_residual_exists hw S y)

theorem wResidual_eq_of_space_eq (w : I → ℝ) (hw : ∀ i, 0 < w i)
    {S T : Submodule ℝ (I → ℝ)} (hST : S = T) (y : I → ℝ) :
    wResidual w hw S y = wResidual w hw T y := by
  apply residual_unique hw (wResidual_spec w hw S y)
  rw [hST]
  exact wResidual_spec w hw T y

theorem wResidual_mem_zero (w : I → ℝ) (hw : ∀ i, 0 < w i)
    {S : Submodule ℝ (I → ℝ)} {y : I → ℝ} (hy : y ∈ S) :
    wResidual w hw S y = 0 := by
  apply residual_unique hw (wResidual_spec w hw S y)
  exact ⟨by simpa using hy, fun z _ => by simp [wInner]⟩

theorem wResidual_add (w : I → ℝ) (hw : ∀ i, 0 < w i)
    (S : Submodule ℝ (I → ℝ)) (x y : I → ℝ) :
    wResidual w hw S (x + y) = wResidual w hw S x + wResidual w hw S y := by
  have hx := wResidual_spec w hw S x
  have hy := wResidual_spec w hw S y
  apply residual_unique hw (wResidual_spec w hw S (x + y))
  constructor
  · convert S.add_mem hx.1 hy.1 using 1 <;> abel
  · intro z hz
    rw [wInner_add_left, hx.2 z hz, hy.2 z hz, add_zero]

theorem wResidual_smul (w : I → ℝ) (hw : ∀ i, 0 < w i)
    (S : Submodule ℝ (I → ℝ)) (a : ℝ) (y : I → ℝ) :
    wResidual w hw S (a • y) = a • wResidual w hw S y := by
  have hy := wResidual_spec w hw S y
  apply residual_unique hw (wResidual_spec w hw S (a • y))
  constructor
  · simpa only [smul_sub] using S.smul_mem a hy.1
  · intro z hz
    rw [wInner_smul_left, hy.2 z hz, mul_zero]

/-- The actual weighted within operator, constructed from existence and uniqueness. -/
def wResidualMap (w : I → ℝ) (hw : ∀ i, 0 < w i)
    (S : Submodule ℝ (I → ℝ)) : (I → ℝ) →ₗ[ℝ] (I → ℝ) where
  toFun := wResidual w hw S
  map_add' := wResidual_add w hw S
  map_smul' := wResidual_smul w hw S

end EconHDFE
