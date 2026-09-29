import Mathlib

/-!
# Finite positive diagonal-weight least squares

Feasibility and normal equations are both required. Their equivalence to
minimization and uniqueness are proved, not assumed. This concerns exact real
arithmetic, not convergence or floating-point implementation correctness.
-/
namespace EconHDFE
open scoped BigOperators
variable {I : Type*} [Fintype I]

def wInner (w x z : I → ℝ) : ℝ := ∑ i, w i * x i * z i

theorem wInner_comm (w x z : I → ℝ) :
    wInner w x z = wInner w z x := by
  unfold wInner
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem wInner_add_left (w x y z : I → ℝ) :
    wInner w (x + y) z = wInner w x z + wInner w y z := by
  simp only [wInner, Pi.add_apply, mul_add, add_mul, Finset.sum_add_distrib]

theorem wInner_add_right (w x y z : I → ℝ) :
    wInner w x (y + z) = wInner w x y + wInner w x z := by
  simp only [wInner, Pi.add_apply, mul_add, Finset.sum_add_distrib]

theorem wInner_sub_left (w x y z : I → ℝ) :
    wInner w (x - y) z = wInner w x z - wInner w y z := by
  simp only [wInner, Pi.sub_apply, mul_sub, sub_mul, Finset.sum_sub_distrib]

theorem wInner_sub_right (w x y z : I → ℝ) :
    wInner w x (y - z) = wInner w x y - wInner w x z := by
  simp only [wInner, Pi.sub_apply, mul_sub, Finset.sum_sub_distrib]

theorem wInner_smul_left (w x z : I → ℝ) (a : ℝ) :
    wInner w (a • x) z = a * wInner w x z := by
  simp only [wInner, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem wInner_smul_right (w x z : I → ℝ) (a : ℝ) :
    wInner w x (a • z) = a * wInner w x z := by
  rw [wInner_comm, wInner_smul_left, wInner_comm w z x]

theorem wInner_sub_smul_self (w u z : I → ℝ) (t : ℝ) :
    wInner w (u - t • z) (u - t • z) =
    wInner w u u - 2 * t * wInner w u z + t ^ 2 * wInner w z z := by
  simp only [wInner_sub_left, wInner_sub_right,
    wInner_smul_right, wInner_smul_left, wInner_comm w z u]
  ring

theorem wInner_self_nonneg {w : I → ℝ} (hw : ∀ i, 0 ≤ w i) (x : I → ℝ) :
    0 ≤ wInner w x x := by
  apply Finset.sum_nonneg
  intro i _
  simpa [pow_two, mul_assoc] using mul_nonneg (hw i) (sq_nonneg (x i))

theorem wInner_self_eq_zero {w x : I → ℝ} (hw : ∀ i, 0 < w i)
    (h : wInner w x x = 0) : x = 0 := by
  classical
  ext i
  have term_nonneg : ∀ j : I, 0 ≤ w j * x j * x j := by
    intro j
    simpa [pow_two, mul_assoc] using mul_nonneg (le_of_lt (hw j)) (sq_nonneg (x j))
  have bound : w i * x i * x i ≤ ∑ j, w j * x j * x j :=
    Finset.single_le_sum (f := fun j => w j * x j * x j)
      (fun j _ => term_nonneg j) (Finset.mem_univ i)
  change _ ≤ wInner w x x at bound
  rw [h] at bound
  by_contra hn
  have hp : 0 < w i * (x i * x i) :=
    mul_pos (hw i) (mul_self_pos.mpr hn)
  nlinarith

def IsResidual (w : I → ℝ) (S : Submodule ℝ (I → ℝ)) (y r : I → ℝ) : Prop :=
  y - r ∈ S ∧ ∀ z ∈ S, wInner w r z = 0

def IsWLSFit (w : I → ℝ) (S : Submodule ℝ (I → ℝ)) (y f : I → ℝ) : Prop :=
  f ∈ S ∧ ∀ g ∈ S, wInner w (y - f) (y - f) ≤ wInner w (y - g) (y - g)

theorem residual_unique {w : I → ℝ} (hw : ∀ i, 0 < w i)
    {S : Submodule ℝ (I → ℝ)} {y r s : I → ℝ}
    (hr : IsResidual w S y r) (hs : IsResidual w S y s) : r = s := by
  have hdiff : r - s ∈ S := by
    have hh := S.sub_mem hs.1 hr.1
    convert hh using 1
    abel
  have hz : wInner w (r - s) (r - s) = 0 := by
    rw [wInner_sub_left, hr.2 _ hdiff, hs.2 _ hdiff, sub_self]
  exact sub_eq_zero.mp (wInner_self_eq_zero hw hz)

theorem residual_isWLSFit {w : I → ℝ} (hw : ∀ i, 0 ≤ w i)
    {S : Submodule ℝ (I → ℝ)} {y r : I → ℝ} (hr : IsResidual w S y r) :
    IsWLSFit w S y (y - r) := by
  refine ⟨hr.1, ?_⟩
  intro g hg
  have hd : (y - r) - g ∈ S := S.sub_mem hr.1 hg
  have hn := wInner_self_nonneg hw ((y - r) - g)
  have hleft : y - (y - r) = r := by abel
  have hright : y - g = r + ((y - r) - g) := by abel
  rw [hleft, hright, wInner_add_left, wInner_add_right, wInner_add_right,
    hr.2 _ hd, wInner_comm w ((y - r) - g) r, hr.2 _ hd]
  linarith

theorem wlsFit_isResidual {w : I → ℝ} (hw : ∀ i, 0 < w i)
    {S : Submodule ℝ (I → ℝ)} {y f : I → ℝ}
    (hf : IsWLSFit w S y f) : IsResidual w S y (y - f) := by
  constructor
  · simpa only [sub_sub_cancel] using hf.1
  · intro z hz
    let a := wInner w (y - f) z
    let b := wInner w z z
    have hb : 0 ≤ b := wInner_self_nonneg (fun i => le_of_lt (hw i)) z
    by_cases hb0 : b = 0
    · have hz0 : z = 0 := wInner_self_eq_zero hw hb0
      simp [wInner, hz0]
    · have hbpos : 0 < b := lt_of_le_of_ne hb (Ne.symm hb0)
      have hg := hf.2 (f + (a / b) • z) (S.add_mem hf.1 (S.smul_mem _ hz))
      have rearrange : y - (f + (a / b) • z) = (y - f) - (a / b) • z := by abel
      rw [rearrange] at hg
      have expansion :
          wInner w ((y - f) - (a / b) • z) ((y - f) - (a / b) • z) =
          wInner w (y - f) (y - f) - a * a / b := by
        rw [wInner_sub_smul_self]
        change wInner w (y - f) (y - f) - 2 * (a / b) * a + (a / b) ^ 2 * b =
          wInner w (y - f) (y - f) - a * a / b
        field_simp
        <;> ring
      rw [expansion] at hg
      have hab : a * a / b ≤ 0 := by linarith
      have hmul := mul_nonpos_of_nonpos_of_nonneg hab (le_of_lt hbpos)
      have cancel : a * a / b * b = a * a := by field_simp
      rw [cancel] at hmul
      change a = 0
      nlinarith [sq_nonneg a]

theorem wlsFit_iff_residual {w : I → ℝ} (hw : ∀ i, 0 < w i)
    {S : Submodule ℝ (I → ℝ)} {y f : I → ℝ} :
    IsWLSFit w S y f ↔ IsResidual w S y (y - f) := by
  constructor
  · exact wlsFit_isResidual hw
  · intro h
    simpa only [sub_sub_cancel] using
      residual_isWLSFit (fun i => le_of_lt (hw i)) h

/-- Identical sample, outcome and positive diagonal weights. -/
theorem wls_fitted_invariance {w : I → ℝ} (hw : ∀ i, 0 < w i)
    {S T : Submodule ℝ (I → ℝ)} (hST : S = T)
    {y f g : I → ℝ} (hf : IsWLSFit w S y f) (hg : IsWLSFit w T y g) :
    f = g := by
  subst T
  have h := residual_unique hw (wlsFit_isResidual hw hf) (wlsFit_isResidual hw hg)
  exact sub_right_injective h

/-- A genuine singleton column forces its observation's residual to vanish. -/
theorem singleton_residual_zero [DecidableEq I] {w : I → ℝ}
    (hw : ∀ i, 0 < w i) {S : Submodule ℝ (I → ℝ)}
    {y r : I → ℝ} (hr : IsResidual w S y r)
    (i : I) (hi : Pi.single i (1 : ℝ) ∈ S) : r i = 0 := by
  have h := hr.2 _ hi
  simp [wInner, Pi.single_apply, mul_ite] at h
  exact h.resolve_left (ne_of_gt (hw i))

end EconHDFE
