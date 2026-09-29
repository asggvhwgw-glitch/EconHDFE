import Mathlib

/-!
# Finite PPML optima under exact design-space reductions

The objective is the actual weighted Poisson negative log-likelihood up to
terms independent of the predictor. Positive weights make it strictly convex
in the finite predictor. Existence of a finite optimum is not asserted: this
result does not remove separation or prove iterative convergence.
-/
noncomputable section
namespace EconHDFE
open scoped Classical BigOperators
variable {I : Type*} [Fintype I]

def ppmlLoss (w y eta : I → ℝ) : ℝ :=
  ∑ i, w i * (Real.exp (eta i) - y i * eta i)

def predictorMidpoint (eta theta : I → ℝ) : I → ℝ :=
  fun i => (1 / 2 : ℝ) * eta i + (1 / 2 : ℝ) * theta i

def IsPPMLFit (w y offset : I → ℝ) (S : Submodule ℝ (I → ℝ)) (eta : I → ℝ) : Prop :=
  eta - offset ∈ S ∧
    ∀ theta, theta - offset ∈ S → ppmlLoss w y eta ≤ ppmlLoss w y theta

theorem poisson_midpoint_le (w y a b : ℝ) (hw : 0 ≤ w) :
    w * (Real.exp ((1 / 2 : ℝ) * a + (1 / 2 : ℝ) * b) -
      y * ((1 / 2 : ℝ) * a + (1 / 2 : ℝ) * b)) ≤
    (1 / 2 : ℝ) * (w * (Real.exp a - y * a)) +
      (1 / 2 : ℝ) * (w * (Real.exp b - y * b)) := by
  have he : Real.exp ((1 / 2 : ℝ) * a + (1 / 2 : ℝ) * b) ≤
      (1 / 2 : ℝ) * Real.exp a + (1 / 2 : ℝ) * Real.exp b :=
    convexOn_exp.2 (Set.mem_univ a) (Set.mem_univ b) (by norm_num) (by norm_num) (by norm_num)
  have hm := mul_le_mul_of_nonneg_left he hw
  nlinarith

theorem poisson_midpoint_lt (w y a b : ℝ) (hw : 0 < w) (hab : a ≠ b) :
    w * (Real.exp ((1 / 2 : ℝ) * a + (1 / 2 : ℝ) * b) -
      y * ((1 / 2 : ℝ) * a + (1 / 2 : ℝ) * b)) <
    (1 / 2 : ℝ) * (w * (Real.exp a - y * a)) +
      (1 / 2 : ℝ) * (w * (Real.exp b - y * b)) := by
  have he : Real.exp ((1 / 2 : ℝ) * a + (1 / 2 : ℝ) * b) <
      (1 / 2 : ℝ) * Real.exp a + (1 / 2 : ℝ) * Real.exp b :=
    strictConvexOn_exp.2 (Set.mem_univ a) (Set.mem_univ b) hab
      (by norm_num) (by norm_num) (by norm_num)
  have hm := mul_lt_mul_of_pos_left he hw
  nlinarith

/-- Strictness follows from at least one differing coordinate and positive weights. -/
theorem ppml_midpoint_strict {w : I → ℝ} (hw : ∀ i, 0 < w i)
    (y eta theta : I → ℝ) (hne : eta ≠ theta) :
    ppmlLoss w y (predictorMidpoint eta theta) <
      (1 / 2 : ℝ) * ppmlLoss w y eta + (1 / 2 : ℝ) * ppmlLoss w y theta := by
  obtain ⟨j, hj⟩ := Function.ne_iff.mp hne
  have hsum := Finset.sum_lt_sum
    (s := (Finset.univ : Finset I))
    (f := fun i => w i * (Real.exp ((1 / 2 : ℝ) * eta i + (1 / 2 : ℝ) * theta i) -
      y i * ((1 / 2 : ℝ) * eta i + (1 / 2 : ℝ) * theta i)))
    (g := fun i => (1 / 2 : ℝ) * (w i * (Real.exp (eta i) - y i * eta i)) +
      (1 / 2 : ℝ) * (w i * (Real.exp (theta i) - y i * theta i)))
    (fun i _ => poisson_midpoint_le (w i) (y i) (eta i) (theta i) (hw i).le)
    ⟨j, Finset.mem_univ j, poisson_midpoint_lt (w j) (y j) (eta j) (theta j) (hw j) hj⟩
  simpa only [ppmlLoss, predictorMidpoint, Finset.sum_add_distrib, ← Finset.mul_sum] using hsum

theorem predictor_midpoint_feasible (S : Submodule ℝ (I → ℝ))
    (offset eta theta : I → ℝ) (he : eta - offset ∈ S) (ht : theta - offset ∈ S) :
    predictorMidpoint eta theta - offset ∈ S := by
  have hm := S.add_mem (S.smul_mem (1 / 2 : ℝ) he) (S.smul_mem (1 / 2 : ℝ) ht)
  convert hm using 1
  ext i
  simp only [predictorMidpoint, Pi.sub_apply, Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  ring

/-- Uniqueness is proved for the predictor, not for redundant coefficients. -/
theorem finite_ppml_predictor_unique {w : I → ℝ} (hw : ∀ i, 0 < w i)
    {y offset : I → ℝ} {S : Submodule ℝ (I → ℝ)} {eta theta : I → ℝ}
    (he : IsPPMLFit w y offset S eta) (ht : IsPPMLFit w y offset S theta) : eta = theta := by
  by_contra hne
  have hm := predictor_midpoint_feasible S offset eta theta he.1 ht.1
  have hstrict := ppml_midpoint_strict hw y eta theta hne
  have ha := he.2 _ hm
  have hb := ht.2 _ hm
  linarith

/-- Same sample, objective, offset, weights and feasible predictor set. -/
theorem ppml_problem_invariance (w y offset : I → ℝ)
    {S T : Submodule ℝ (I → ℝ)} (hST : S = T) (eta : I → ℝ) :
    IsPPMLFit w y offset S eta ↔ IsPPMLFit w y offset T eta := by rw [hST]

/-- Equal design spaces and attained finite optima imply equal fitted predictors. -/
theorem finite_ppml_fitted_invariance {w : I → ℝ} (hw : ∀ i, 0 < w i)
    {y offset : I → ℝ} {S T : Submodule ℝ (I → ℝ)} (hST : S = T)
    {eta theta : I → ℝ} (he : IsPPMLFit w y offset S eta)
    (ht : IsPPMLFit w y offset T theta) : eta = theta := by
  rw [← hST] at ht
  exact finite_ppml_predictor_unique hw he ht

theorem finite_ppml_response_invariance {w : I → ℝ} (hw : ∀ i, 0 < w i)
    {y offset : I → ℝ} {S T : Submodule ℝ (I → ℝ)} (hST : S = T)
    {eta theta : I → ℝ} (he : IsPPMLFit w y offset S eta)
    (ht : IsPPMLFit w y offset T theta) :
    (fun i => Real.exp (eta i)) = (fun i => Real.exp (theta i)) := by
  rw [finite_ppml_fitted_invariance hw hST he ht]

end EconHDFE
