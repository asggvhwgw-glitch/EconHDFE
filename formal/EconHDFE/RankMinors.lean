import EconHDFE.RankBasics

/-!
# Actual rank and nonzero minors

Select a basis from the original columns, then from its rows. This proves
existence of a maximal nonzero minor, including the rank-zero case. The
existence result is mathematical, not an executable minor-search algorithm.
-/
noncomputable section
namespace EconHDFE
open scoped Classical
variable {F L I V : Type*} [Field F] [Fintype I] [Fintype V]

/-- A nonzero square minor lower-bounds the actual matrix rank. -/
theorem rank_lower_of_nonzero_minor {r : ℕ} (A : Matrix I V F)
    (rows : Fin r → I) (cols : Fin r → V)
    (hdet : (A.submatrix rows cols).det ≠ 0) : r ≤ A.rank := by
  have hunit : IsUnit (A.submatrix rows cols) :=
    Matrix.isUnit_iff_isUnit_det.mpr (isUnit_iff_ne_zero.mpr hdet)
  have hr : (A.submatrix rows cols).rank = r := by
    simpa only [Fintype.card_fin] using
      Matrix.rank_of_isUnit (A.submatrix rows cols) hunit
  calc
    r = (A.submatrix rows cols).rank := hr.symm
    _ ≤ A.rank := matrix_rank_submatrix_le A rows cols

/-- Full rank of a finite square matrix implies nonzero determinant. -/
theorem square_det_nonzero_of_rank {n : Type*} [Fintype n]
    (A : Matrix n n F) (hr : A.rank = Fintype.card n) : A.det ≠ 0 := by
  have htop : LinearMap.range A.mulVecLin = ⊤ :=
    Submodule.eq_top_of_finrank_eq (by simpa [Matrix.rank] using hr)
  have hinj : Function.Injective A.mulVecLin :=
    LinearMap.injective_iff_surjective.mpr (LinearMap.range_eq_top.mp htop)
  intro hz
  obtain ⟨x, hx, hAx⟩ := Matrix.exists_mulVec_eq_zero_iff.mpr hz
  apply hx
  apply hinj
  simpa using hAx

/-- Choose actual original columns spanning the same image, without repetition. -/
theorem matrix_exists_rank_columns (A : Matrix I V F) (r : ℕ) (hr : A.rank = r) :
    ∃ cols : Fin r → V, Function.Injective cols ∧
      (A.submatrix id cols).rank = r := by
  subst r
  obtain ⟨J, a, ha, hspan, hli⟩ := exists_linearIndependent' F A.col
  letI : Fintype J := Fintype.ofInjective a ha
  have hc : Fintype.card J = A.rank := by
    rw [Matrix.rank_eq_finrank_span_cols, ← hspan]
    exact linearIndependent_iff_card_eq_finrank_span.mp hli
  let e : J ≃ Fin A.rank := Fintype.equivFinOfCardEq hc
  let cols : Fin A.rank → V := a ∘ e.symm
  refine ⟨cols, ha.comp e.symm.injective, ?_⟩
  have hsets : Set.range (A.col ∘ cols) = Set.range (A.col ∘ a) := by
    ext z
    constructor
    · rintro ⟨j, rfl⟩
      exact ⟨e.symm j, rfl⟩
    · rintro ⟨j, rfl⟩
      exact ⟨e j, by simp [cols]⟩
  rw [Matrix.rank_eq_finrank_span_cols, Matrix.rank_eq_finrank_span_cols]
  change Module.finrank F (Submodule.span F (Set.range (A.col ∘ cols))) = _
  rw [hsets, hspan]

/-- A matrix has a nonzero minor of order exactly its actual rank. -/
theorem matrix_exists_rank_minor (A : Matrix I V F) :
    ∃ (rows : Fin A.rank → I) (cols : Fin A.rank → V),
      Function.Injective rows ∧ Function.Injective cols ∧
        (A.submatrix rows cols).det ≠ 0 := by
  obtain ⟨cols, hcols, hc⟩ := matrix_exists_rank_columns A A.rank rfl
  obtain ⟨rows, hrows, hr⟩ := matrix_exists_rank_columns
    (A.submatrix id cols).transpose A.rank ((Matrix.rank_transpose _).trans hc)
  refine ⟨rows, cols, hrows, hcols, square_det_nonzero_of_rank _ ?_⟩
  have heq : (A.submatrix rows cols).transpose =
      (A.submatrix id cols).transpose.submatrix id rows := rfl
  rw [← Matrix.rank_transpose, heq]
  simpa only [Fintype.card_fin] using hr

/-- Taking a minor commutes with determinant under a ring homomorphism. -/
theorem minor_det_map {R S : Type*} [CommRing R] [CommRing S] {r : ℕ}
    (f : R →+* S) (A : Matrix I V R) (rows : Fin r → I) (cols : Fin r → V) :
    ((A.map f).submatrix rows cols).det = f ((A.submatrix rows cols).det) := by
  exact (RingHom.map_det f (A.submatrix rows cols)).symm

/-- Reflect a maximal target minor to bound rank after mapping a field. -/
theorem matrix_rank_map_le [Field L] (f : F →+* L) (A : Matrix I V F) :
    (A.map f).rank ≤ A.rank := by
  obtain ⟨rows, cols, _, _, hd⟩ := matrix_exists_rank_minor (A.map f)
  apply rank_lower_of_nonzero_minor A rows cols
  intro hz
  apply hd
  rw [minor_det_map, hz, map_zero]

/-- A field homomorphism preserves the actual rank, not just zero patterns. -/
theorem matrix_rank_map_field [Field L] (f : F →+* L) (A : Matrix I V F) :
    (A.map f).rank = A.rank := by
  apply le_antisymm (matrix_rank_map_le f A)
  obtain ⟨rows, cols, _, _, hd⟩ := matrix_exists_rank_minor A
  apply rank_lower_of_nonzero_minor (A.map f) rows cols
  rw [minor_det_map]
  intro hz
  exact hd (f.injective (by simpa only [map_zero] using hz))

end EconHDFE
