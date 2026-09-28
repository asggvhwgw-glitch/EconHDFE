import EconHDFE.CategoricalDesign

/-!
# Actual matrix rank and duplicate-row reduction

All ranks are dimensions of actual matrix images. Equal nonzero patterns alone
are not a rank certificate. The field-generic reductions apply independently
over rationals, reals, or a finite field; they do not assert equality between
ranks in different characteristics.
-/
noncomputable section
namespace EconHDFE
open scoped Classical BigOperators
variable {F I J V U G : Type*} {K : G → Type*}

/-- The actual 0/1 incidence matrix, not a generic sparsity-pattern rank. -/
def incidenceMatrix (F : Type*) [Zero F] [One F]
    (code : (g : G) → I → K g) : Matrix I (Sigma K) F :=
  fun i v => if code v.1 i = v.2 then 1 else 0

/-- Connect matrix rank to the categorical space used in the projection proofs. -/
theorem incidence_rank_eq_finrank [Fintype G] [∀ g, Fintype (K g)]
    (code : (g : G) → I → K g) :
    (incidenceMatrix ℝ code).rank = Module.finrank ℝ (feSpace code) := by
  rw [Matrix.rank_eq_finrank_span_cols]
  rfl

/-- Actual equality of row sets preserves rank; row multiplicities do not matter. -/
theorem matrix_rank_of_same_rows [Field F] [Fintype I] [Fintype J] [Fintype V]
    (A : Matrix I V F) (B : Matrix J V F) (h : Set.range A = Set.range B) :
    A.rank = B.rank := by
  rw [Matrix.rank_eq_finrank_span_row, Matrix.rank_eq_finrank_span_row]
  exact congrArg (fun s : Set (V → F) => Module.finrank F (Submodule.span F s)) h

/-- Representatives must cover every original row value, not every row identity. -/
theorem matrix_rank_dedup [Field F] [Fintype I] [Fintype J] [Fintype V]
    (A : Matrix I V F) (rows : J → I)
    (cover : ∀ i, ∃ j, A i = A (rows j)) :
    A.rank = (A.submatrix rows id).rank := by
  apply matrix_rank_of_same_rows
  ext v
  constructor
  · rintro ⟨i, rfl⟩
    obtain ⟨j, hj⟩ := cover i
    exact ⟨j, hj.symm⟩
  · rintro ⟨j, rfl⟩
    exact ⟨rows j, rfl⟩

/-- Corrected manuscript lem:dedup, with explicit coverage of categorical tuples. -/
theorem categorical_rank_dedup [Field F] [Fintype I] [Fintype J]
    [Fintype G] [∀ g, Fintype (K g)]
    (code : (g : G) → I → K g) (rows : J → I)
    (cover : ∀ i, ∃ j, ∀ g, code g i = code g (rows j)) :
    (incidenceMatrix F code).rank =
      (incidenceMatrix F (fun g j => code g (rows j))).rank := by
  apply matrix_rank_dedup (incidenceMatrix F code) rows
  intro i
  obtain ⟨j, hj⟩ := cover i
  refine ⟨j, ?_⟩
  funext v
  simp only [incidenceMatrix, hj v.1]

/-- Rectangular row restriction, including repeated or reordered selections. -/
theorem matrix_rank_rows_le [Field F] [Fintype I] [Fintype J] [Fintype V]
    (A : Matrix I V F) (rows : J → I) :
    (A.submatrix rows id).rank ≤ A.rank := by
  rw [Matrix.rank_eq_finrank_span_row, Matrix.rank_eq_finrank_span_row]
  apply Submodule.finrank_mono
  apply Submodule.span_mono
  rintro _ ⟨j, rfl⟩
  exact ⟨rows j, rfl⟩

/-- Rectangular column restriction cannot increase actual rank. -/
theorem matrix_rank_columns_le [Field F] [Fintype I] [Fintype V] [Fintype U]
    (A : Matrix I V F) (cols : U → V) :
    (A.submatrix id cols).rank ≤ A.rank := by
  rw [Matrix.rank_eq_finrank_span_cols, Matrix.rank_eq_finrank_span_cols]
  apply Submodule.finrank_mono
  apply Submodule.span_mono
  rintro _ ⟨j, rfl⟩
  exact ⟨cols j, rfl⟩

/-- General rectangular minor bound, independent of sparsity conventions. -/
theorem matrix_rank_submatrix_le [Field F] [Fintype I] [Fintype J]
    [Fintype V] [Fintype U] (A : Matrix I V F) (rows : J → I) (cols : U → V) :
    (A.submatrix rows cols).rank ≤ A.rank :=
  (matrix_rank_columns_le (A.submatrix rows id) cols).trans (matrix_rank_rows_le A rows)

end EconHDFE
