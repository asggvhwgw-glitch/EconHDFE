import EconHDFE.RankBasics

/-!
# Mathematical consequences for absorbed degrees of freedom

Prefix allocation is finite-dimensional algebra, not verification of a rank
routine. Nonzero row rescaling preserves the actual matrix rank; the positive
square-root weight corollary explains the WLS rank target.
-/
noncomputable section
namespace EconHDFE
open scoped Classical BigOperators
variable {F I V : Type*} [Field F] [Fintype I] [Fintype V]

theorem nonzero_diagonal_rank (A : Matrix I V F) (w : I → F)
    (hw : ∀ i, w i ≠ 0) : (Matrix.diagonal w * A).rank = A.rank := by
  apply Matrix.rank_mul_eq_right_of_isUnit_det
  apply isUnit_iff_ne_zero.mpr
  rw [Matrix.det_diagonal]
  exact Finset.prod_ne_zero_iff.mpr (fun i _ => hw i)

theorem positive_weight_sqrt_rank (A : Matrix I V ℝ) (w : I → ℝ)
    (hw : ∀ i, 0 < w i) :
    (Matrix.diagonal (fun i => Real.sqrt (w i)) * A).rank = A.rank :=
  nonzero_diagonal_rank A _ (fun i => ne_of_gt (Real.sqrt_pos.2 (hw i)))

def prefixSpace (blocks : ℕ → Submodule F (I → F)) : ℕ → Submodule F (I → F)
  | 0 => ⊥
  | n + 1 => prefixSpace blocks n ⊔ blocks n

def prefixDimension (blocks : ℕ → Submodule F (I → F)) (n : ℕ) : ℕ :=
  Module.finrank F (prefixSpace blocks n)

theorem prefix_dimension_step (blocks : ℕ → Submodule F (I → F)) (n : ℕ) :
    prefixDimension blocks n ≤ prefixDimension blocks (n + 1) :=
  Submodule.finrank_mono (show prefixSpace blocks n ≤ prefixSpace blocks (n + 1)
    from le_sup_left)

theorem prefix_increment_le (blocks : ℕ → Submodule F (I → F)) (n : ℕ) :
    prefixDimension blocks (n + 1) - prefixDimension blocks n ≤
      Module.finrank F (blocks n) := by
  have h := Submodule.finrank_add_le_finrank_add_finrank (prefixSpace blocks n) (blocks n)
  change prefixDimension blocks (n + 1) ≤
    prefixDimension blocks n + Module.finrank F (blocks n) at h
  omega

/-- The increments telescope exactly; no rank is obtained from a reporting convention. -/
theorem prefix_increment_total (blocks : ℕ → Submodule F (I → F)) (n : ℕ) :
    (∑ j ∈ Finset.range n, (prefixDimension blocks (j + 1) - prefixDimension blocks j)) =
      prefixDimension blocks n := by
  induction n with
  | zero => simp [prefixDimension, prefixSpace]
  | succ n ih =>
    rw [Finset.sum_range_succ, ih]
    have h := prefix_dimension_step blocks n
    omega

/-- Correct redundancy accounting when each block width bounds its own dimension. -/
theorem prefix_redundancy_total (blocks : ℕ → Submodule F (I → F))
    (width : ℕ → ℕ) (n : ℕ)
    (bound : ∀ j < n, Module.finrank F (blocks j) ≤ width j) :
    (∑ j ∈ Finset.range n,
      (width j - (width j - (prefixDimension blocks (j + 1) - prefixDimension blocks j)))) =
      prefixDimension blocks n := by
  rw [← prefix_increment_total blocks n]
  apply Finset.sum_congr rfl
  intro j hj
  have h := (prefix_increment_le blocks j).trans (bound j (Finset.mem_range.mp hj))
  omega

/-- Specialization to actual matrix blocks supplies the width bound, rather than assuming it. -/
theorem matrix_prefix_redundancy_total (width : ℕ → ℕ)
    (blocks : (j : ℕ) → Matrix I (Fin (width j)) F) (n : ℕ) :
    (∑ j ∈ Finset.range n,
      (width j - (width j -
        (prefixDimension (fun k => LinearMap.range (blocks k).mulVecLin) (j + 1) -
         prefixDimension (fun k => LinearMap.range (blocks k).mulVecLin) j)))) =
      prefixDimension (fun k => LinearMap.range (blocks k).mulVecLin) n := by
  apply prefix_redundancy_total
  intro j _
  change (blocks j).rank ≤ width j
  simpa only [Fintype.card_fin] using (blocks j).rank_le_card_width

/-- Reindexing all columns leaves total rank unchanged, regardless of block ordering. -/
theorem rank_column_permutation (A : Matrix I V F) (e : V ≃ V) :
    (A.submatrix id e).rank = A.rank := by
  exact Matrix.rank_submatrix A (Equiv.refl I) e

end EconHDFE
