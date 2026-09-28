import EconHDFE.RankBasics

/-!
# Rank additivity for actual separated incidence components

The block sizes may differ. A row/column reindexing and zero cross-component
entries suffice; a graph routine's proposed component labels are not assumed
correct merely because they are called components.
-/
noncomputable section
namespace EconHDFE
open scoped Classical BigOperators
variable {F H I V : Type*} [Field F]
variable {R C : H → Type*} [Fintype H] [∀ h, Fintype (R h)] [∀ h, Fintype (C h)]

/-- Multiplication by an actual block diagonal matrix separates by component. -/
theorem component_mulVec (A : ∀ h, Matrix (R h) (C h) F)
    (a : Sigma C → F) (h : H) (i : R h) :
    (Matrix.blockDiagonal' A).mulVec a ⟨h, i⟩ =
      (A h).mulVec (fun j => a ⟨h, j⟩) i := by
  simp only [Matrix.mulVec, dotProduct, ← Finset.univ_sigma_univ, Finset.sum_sigma]
  rw [Fintype.sum_eq_single h]
  · simp only [Matrix.blockDiagonal'_apply_eq]
  · intro k hk
    apply Finset.sum_eq_zero
    intro j _
    rw [Matrix.blockDiagonal'_apply_ne A i j (Ne.symm hk), zero_mul]

/-- Explicit linear equivalence, rather than an assumed dimension identity. -/
def componentRangeEquiv (A : ∀ h, Matrix (R h) (C h) F) :
    LinearMap.range (Matrix.blockDiagonal' A).mulVecLin ≃ₗ[F]
      ((h : H) → LinearMap.range (A h).mulVecLin) where
  toFun x h := ⟨fun i => x.val ⟨h, i⟩, by
    obtain ⟨a, ha⟩ := x.property
    refine ⟨fun j => a ⟨h, j⟩, ?_⟩
    funext i
    calc
      (A h).mulVec (fun j => a ⟨h, j⟩) i =
          (Matrix.blockDiagonal' A).mulVec a ⟨h, i⟩ := (component_mulVec A a h i).symm
      _ = x.val ⟨h, i⟩ := congrFun ha ⟨h, i⟩⟩
  invFun x := ⟨fun i => (x i.1).val i.2, by
    choose a ha using fun h => (x h).property
    refine ⟨fun j => a j.1 j.2, ?_⟩
    funext i
    rcases i with ⟨h, i⟩
    change (Matrix.blockDiagonal' A).mulVec (fun j => a j.1 j.2) ⟨h, i⟩ = (x h).val i
    rw [component_mulVec]
    exact congrFun (ha h) i⟩
  left_inv x := by
    apply Subtype.ext
    funext i
    cases i
    rfl
  right_inv x := by
    funext h
    apply Subtype.ext
    rfl
  map_add' x y := rfl
  map_smul' a x := rfl

/-- Any finite number of rectangular blocks, including empty blocks. -/
theorem matrix_rank_block_diagonal (A : ∀ h, Matrix (R h) (C h) F) :
    (Matrix.blockDiagonal' A).rank = ∑ h, (A h).rank := by
  calc
    (Matrix.blockDiagonal' A).rank =
        Module.finrank F ((h : H) → LinearMap.range (A h).mulVecLin) :=
      (componentRangeEquiv A).finrank_eq
    _ = ∑ h, (A h).rank := by
      rw [Module.finrank_pi_fintype]
      rfl

/-- Corrected manuscript lem:components: actual zero blocks are checked entrywise. -/
theorem matrix_rank_components [Fintype I] [Fintype V]
    (A : Matrix I V F) (rows : I ≃ Sigma R) (cols : V ≃ Sigma C)
    (separate : ∀ i j, (rows i).1 ≠ (cols j).1 → A i j = 0) :
    A.rank = ∑ h,
      (A.submatrix (fun i : R h => rows.symm ⟨h, i⟩)
        (fun j : C h => cols.symm ⟨h, j⟩)).rank := by
  let blocks := fun h => A.submatrix (fun i : R h => rows.symm ⟨h, i⟩)
    (fun j : C h => cols.symm ⟨h, j⟩)
  have hb : A.reindex rows cols = Matrix.blockDiagonal' blocks := by
    ext ⟨h, i⟩ ⟨k, j⟩
    by_cases hk : h = k
    · subst k
      simp [blocks, Matrix.reindex_apply]
    · rw [Matrix.blockDiagonal'_apply_ne blocks i j hk]
      apply separate
      simpa using hk
  calc
    A.rank = (A.reindex rows cols).rank := (Matrix.rank_reindex rows cols A).symm
    _ = (Matrix.blockDiagonal' blocks).rank := congrArg Matrix.rank hb
    _ = ∑ h, (blocks h).rank := matrix_rank_block_diagonal blocks

end EconHDFE
