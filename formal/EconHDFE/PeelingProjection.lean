import EconHDFE.CategoricalDesign
import EconHDFE.ResidualCore

/-!
# Exact categorical residual-core theorem

Legality of the observation-level trace implies a unit upper-triangular pivot
matrix. Its determinant, surjectivity and reconstruction feasibility are proved
here. The final theorem has no assumed invertible block or assumed equality of
column spaces. This is exact real mathematics, not verification of a queue or
floating-point solver.
-/
noncomputable section
namespace EconHDFE
open scoped BigOperators Classical
variable {I G : Type*} {K : G → Type*} {code : (g : G) → I → K g}

/-- Actual selected rows and pivot columns, in peeling order. -/
def peelMatrix (t : PeelingTrace code) : Matrix (Fin t.length) (Fin t.length) ℝ :=
  fun i j => feColumn code (t.pivot j) (t.row i)

theorem peelMatrix_upper (t : PeelingTrace code) :
    (peelMatrix t).BlockTriangular id := by
  intro i j hij
  exact peeling_column_later_zero t hij

theorem peelMatrix_det_one (t : PeelingTrace code) : (peelMatrix t).det = 1 := by
  rw [Matrix.det_of_upperTriangular (peelMatrix_upper t)]
  simp [peelMatrix, peeling_column_diagonal]

/-- Invertibility is a consequence of trace legality, not an input assumption. -/
theorem peelMatrix_surjective (t : PeelingTrace code) :
    Function.Surjective (fun a => (peelMatrix t).mulVec a) := by
  intro v
  have hunit : IsUnit (peelMatrix t).det := by
    rw [peelMatrix_det_one]
    exact isUnit_one
  refine ⟨((peelMatrix t)⁻¹).mulVec v, ?_⟩
  change (peelMatrix t).mulVec (((peelMatrix t)⁻¹).mulVec v) = v
  rw [Matrix.mulVec_mulVec, Matrix.mul_nonsing_inv _ hunit, Matrix.one_mulVec]

/-- A linear combination of original pivot columns, in original row coordinates. -/
def peelCorrection (t : PeelingTrace code) (a : Fin t.length → ℝ) : I → ℝ :=
  ∑ k, a k • feColumn code (t.pivot k)

theorem peelCorrection_mem (t : PeelingTrace code) (a : Fin t.length → ℝ) :
    peelCorrection t a ∈ feSpace code := by
  apply Submodule.sum_mem
  intro k _
  exact Submodule.smul_mem _ _ (feColumn_mem code (t.pivot k))

theorem peelCorrection_row (t : PeelingTrace code) (a : Fin t.length → ℝ)
    (k : Fin t.length) :
    peelCorrection t a (t.row k) = (peelMatrix t).mulVec a k := by
  simp [peelCorrection, peelMatrix, Matrix.mulVec, dotProduct,
    Finset.sum_apply, Pi.smul_apply, smul_eq_mul, mul_comm]

theorem peelCorrection_core (t : PeelingTrace code) (a : Fin t.length → ℝ)
    (c : PeelingCore t) : peelCorrection t a c.val = 0 := by
  simp [peelCorrection, Finset.sum_apply, Pi.smul_apply, smul_eq_mul,
    peeling_column_core_zero t c]

/-- Every vector supported on removed observations is in the original FE space. -/
theorem off_core_mem (t : PeelingTrace code) (x : I → ℝ)
    (hx : ∀ c : PeelingCore t, x c.val = 0) : x ∈ feSpace code := by
  obtain ⟨a, ha⟩ := peelMatrix_surjective t (fun k => x (t.row k))
  have heq : peelCorrection t a = x := by
    funext i
    by_cases hi : ∃ k, t.row k = i
    · obtain ⟨k, rfl⟩ := hi
      rw [peelCorrection_row]
      exact congrFun ha k
    · have hc : i ∈ surviving t := by
        intro k hk
        exact hi ⟨k, hk⟩
      calc
        peelCorrection t a i = 0 := peelCorrection_core t a ⟨i, hc⟩
        _ = x i := (hx ⟨i, hc⟩).symm
  rw [← heq]
  exact peelCorrection_mem t a

/-- The exact original fitted space is the pullback of the actual core design. -/
theorem peeling_feasible_iff (t : PeelingTrace code) (z : I → ℝ) :
    z ∈ feSpace code ↔ coreRestriction t z ∈ feSpace (coreCode t) := by
  have hmap : (feSpace code).map (coreRestriction t) = feSpace (coreCode t) :=
    feSpace_restrict code (fun c : PeelingCore t => c.val)
  constructor
  · intro hz
    rw [← hmap]
    exact ⟨z, hz, rfl⟩
  · intro hz
    rw [← hmap] at hz
    rcases hz with ⟨x, hx, hxz⟩
    have hzero : ∀ c : PeelingCore t, (z - x) c.val = 0 := by
      intro c
      have heq := congrFun hxz c
      change x c.val = z c.val at heq
      change z c.val - x c.val = 0
      exact sub_eq_zero.mpr heq.symm
    have hsum := (feSpace code).add_mem (off_core_mem t (z - x) hzero) hx
    simpa only [sub_add_cancel] using hsum

/-- Zero insertion uses original row identities, without sorting or deduplication. -/
def peelExtend (t : PeelingTrace code) (r : PeelingCore t → ℝ) : I → ℝ :=
  fun i => if h : i ∈ surviving t then r ⟨i, h⟩ else 0

@[simp]
theorem peelExtend_core (t : PeelingTrace code) (r : PeelingCore t → ℝ)
    (c : PeelingCore t) : peelExtend t r c.val = r c := by
  simp [peelExtend, c.property]

@[simp]
theorem peelExtend_row (t : PeelingTrace code) (r : PeelingCore t → ℝ)
    (k : Fin t.length) : peelExtend t r (t.row k) = 0 := by
  have hi : t.row k ∉ surviving t := fun h => h k rfl
  simp [peelExtend, hi]

/-- A finite diagonal-weight inner product ignores zero coordinates outside a subset. -/
theorem wInner_on_subtype [Fintype I] (p : I → Prop) (w r z : I → ℝ)
    (hr : ∀ i, ¬p i → r i = 0) :
    wInner w r z = wInner (fun i : {i // p i} => w i.val)
      (fun i => r i.val) (fun i => z i.val) := by
  unfold wInner
  calc
    (∑ i, w i * r i * z i) =
        ∑ i ∈ Finset.univ.filter p, w i * r i * z i := by
      symm
      apply Finset.sum_subset (Finset.filter_subset _ _)
      intro i _ hi
      have hni : ¬p i := by simpa using hi
      simp [hr i hni]
    _ = ∑ i : {i // p i}, w i.val * r i.val * z i.val :=
      Finset.sum_subtype _ (by simp) _

/-- Core feasibility and core normal equations imply both full-sample conditions. -/
theorem categorical_residual_lifts [Fintype I] (t : PeelingTrace code)
    {w y : I → ℝ} {rC : PeelingCore t → ℝ}
    (hc : IsResidual (fun c => w c.val) (feSpace (coreCode t))
      (fun c => y c.val) rC) :
    IsResidual w (feSpace code) y (peelExtend t rC) := by
  constructor
  · apply (peeling_feasible_iff t (y - peelExtend t rC)).mpr
    have heq : coreRestriction t (y - peelExtend t rC) =
        (fun c : PeelingCore t => y c.val) - rC := by
      ext c
      simp [coreRestriction, categoricalMap]
    rw [heq]
    exact hc.1
  · intro z hz
    have hzc : (fun c : PeelingCore t => z c.val) ∈ feSpace (coreCode t) :=
      (peeling_feasible_iff t z).mp hz
    calc
      wInner w (peelExtend t rC) z =
          wInner (fun c : PeelingCore t => w c.val) rC (fun c => z c.val) := by
        simpa only [peelExtend_core] using
          wInner_on_subtype (fun i => i ∈ surviving t) w (peelExtend t rC) z
            (fun i hi => by simp [peelExtend, hi])
      _ = 0 := hc.2 _ hzc

/-- Corrected manuscript thm:core, for every legal trace including partial traces. -/
theorem categorical_residual_core [Fintype I] (t : PeelingTrace code)
    {w y r : I → ℝ} (hw : ∀ i, 0 < w i) {rC : PeelingCore t → ℝ}
    (hf : IsResidual w (feSpace code) y r)
    (hc : IsResidual (fun c => w c.val) (feSpace (coreCode t))
      (fun c => y c.val) rC) : r = peelExtend t rC :=
  residual_unique hw hf (categorical_residual_lifts t hc)

/-- A minimizing core fit constructs a minimizing full fit; no full solver is assumed. -/
theorem categorical_core_wls_fit [Fintype I] (t : PeelingTrace code)
    {w y : I → ℝ} (hw : ∀ i, 0 < w i) {fC : PeelingCore t → ℝ}
    (hc : IsWLSFit (fun c => w c.val) (feSpace (coreCode t))
      (fun c => y c.val) fC) :
    IsWLSFit w (feSpace code) y
      (y - peelExtend t ((fun c : PeelingCore t => y c.val) - fC)) :=
  residual_isWLSFit (fun i => le_of_lt (hw i))
    (categorical_residual_lifts t
      (wlsFit_isResidual (fun c : PeelingCore t => hw c.val) hc))

/-- The same trace works for every column and every strictly positive weight update. -/
theorem categorical_multi_rhs [Fintype I] (t : PeelingTrace code) {M : Type*}
    (W Y R : M → I → ℝ) (RC : M → PeelingCore t → ℝ)
    (hw : ∀ m i, 0 < W m i)
    (hf : ∀ m, IsResidual (W m) (feSpace code) (Y m) (R m))
    (hc : ∀ m, IsResidual (fun c => W m c.val) (feSpace (coreCode t))
      (fun c => Y m c.val) (RC m)) :
    ∀ m, R m = peelExtend t (RC m) := by
  intro m
  exact categorical_residual_core t (hw m) (hf m) (hc m)

end EconHDFE
