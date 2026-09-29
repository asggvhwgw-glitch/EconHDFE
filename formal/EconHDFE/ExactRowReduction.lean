import EconHDFE.RankMinors

/-!
# Exact row-reduction traces and normal return

A trace records legitimate replacements by nonzero multiples plus combinations
of retained rows, and removal of zero rows. It contains no assumed equality of
ranks or spans. Completion in an echelon family proves the returned pivot count.
This is mathematical partial correctness, not verification of a Python loop.
-/
noncomputable section
namespace EconHDFE
open scoped Classical BigOperators
variable {F I V : Type*} [Field F]

/-- Cross-multiplication and nonzero normalization preserve the complete row span. -/
theorem span_insert_row_replace (s : Set (V → F)) (row pivot : V → F)
    (a b : F) (hp : pivot ∈ Submodule.span F s) (ha : a ≠ 0) :
    Submodule.span F (insert row s) =
      Submodule.span F (insert (a • row + b • pivot) s) := by
  apply le_antisymm
  · apply Submodule.span_le.mpr
    intro x hx
    rcases Set.mem_insert_iff.mp hx with rfl | hx
    · let S := Submodule.span F (insert (a • row + b • pivot) s)
      have hp' : pivot ∈ S := Submodule.span_mono (Set.subset_insert _ _) hp
      have hn : a • row + b • pivot ∈ S := Submodule.subset_span (Set.mem_insert _ _)
      have hm := S.smul_mem a⁻¹ (S.sub_mem hn (S.smul_mem b hp'))
      simpa [smul_smul, ha] using hm
    · exact Submodule.subset_span (Set.mem_insert_of_mem _ hx)
  · apply Submodule.span_le.mpr
    intro x hx
    rcases Set.mem_insert_iff.mp hx with rfl | hx
    · let S := Submodule.span F (insert row s)
      have hr : row ∈ S := Submodule.subset_span (Set.mem_insert _ _)
      have hp' : pivot ∈ S := Submodule.span_mono (Set.subset_insert _ _) hp
      exact S.add_mem (S.smul_mem a hr) (S.smul_mem b hp')
    · exact Submodule.subset_span (Set.mem_insert_of_mem _ hx)

/-- Nonzero primitive normalization is a special case of the same invariant. -/
theorem span_insert_row_scale (s : Set (V → F)) (row : V → F) (a : F) (ha : a ≠ 0) :
    Submodule.span F (insert row s) = Submodule.span F (insert (a • row) s) := by
  simpa using span_insert_row_replace s row 0 a 0 (Submodule.zero_mem _) ha

/-- Exact cross-multiplication cancels the selected pivot coordinate. -/
theorem cross_elimination_pivot_zero (row pivot : V → F) (c : V) (d : F) :
    ((pivot c / d) • row - (row c / d) • pivot) c = 0 := by
  simp only [Pi.sub_apply, Pi.smul_apply, smul_eq_mul, div_eq_mul_inv]
  ring

/-- A nonzero pivot and nonzero divisor make the cross step rank preserving. -/
theorem cross_elimination_span (s : Set (V → F)) (row pivot : V → F)
    (hp : pivot ∈ Submodule.span F s) (c : V) (hc : pivot c ≠ 0)
    (d : F) (hd : d ≠ 0) :
    Submodule.span F (insert row s) =
      Submodule.span F (insert ((pivot c / d) • row - (row c / d) • pivot) s) := by
  simpa only [neg_smul, sub_eq_add_neg] using
    span_insert_row_replace s row pivot (pivot c / d) (-(row c / d)) hp (div_ne_zero hc hd)

/-- Allowed exact row operations. The pivot must remain in the retained span. -/
inductive ExactRowStep (F : Type*) [Field F] (V : Type*) :
    Set (V → F) → Set (V → F) → Prop where
  | replace (s : Set (V → F)) (row pivot : V → F) (a b : F)
      (hp : pivot ∈ Submodule.span F s) (ha : a ≠ 0) :
      ExactRowStep F V (insert row s) (insert (a • row + b • pivot) s)
  | dropZero (s : Set (V → F)) : ExactRowStep F V (insert 0 s) s

/-- A finite trace of allowed operations, not a supplied rank certificate. -/
inductive ExactRowTrace (F : Type*) [Field F] (V : Type*) :
    Set (V → F) → Set (V → F) → Prop where
  | refl (s : Set (V → F)) : ExactRowTrace F V s s
  | next {s t u : Set (V → F)} (step : ExactRowStep F V s t)
      (rest : ExactRowTrace F V t u) : ExactRowTrace F V s u

theorem exact_row_step_span {s t : Set (V → F)} (step : ExactRowStep F V s t) :
    Submodule.span F s = Submodule.span F t := by
  cases step with
  | replace s row pivot a b hp ha => exact span_insert_row_replace s row pivot a b hp ha
  | dropZero s => simp

theorem exact_row_trace_span {s t : Set (V → F)} (trace : ExactRowTrace F V s t) :
    Submodule.span F s = Submodule.span F t := by
  induction trace with
  | refl => rfl
  | next step rest ih => exact (exact_row_step_span step).trans ih

/-- Distinct echelon pivots supply a triangular nonzero minor, not an assumed rank. -/
theorem echelon_rows_rank [Fintype V] {r : ℕ} (P : Matrix (Fin r) V F)
    (pivots : Fin r → V) (diagonal : ∀ i, P i (pivots i) ≠ 0)
    (earlier_zero : ∀ i j, j < i → P i (pivots j) = 0) : P.rank = r := by
  have hut : (P.submatrix id pivots).BlockTriangular id := by
    intro i j hij
    exact earlier_zero i j hij
  have hd : (P.submatrix id pivots).det ≠ 0 := by
    rw [Matrix.det_of_upperTriangular hut]
    apply Finset.prod_ne_zero_iff.mpr
    intro i _
    exact diagonal i
  have hl := rank_lower_of_nonzero_minor P id pivots hd
  have hu : P.rank ≤ r := by simpa only [Fintype.card_fin] using P.rank_le_card_height
  exact le_antisymm hu hl

/-- Completed exact reduction to echelon rows returns the actual rank. -/
theorem exact_trace_rank [Fintype I] [Fintype V] {r : ℕ}
    (A : Matrix I V F) (P : Matrix (Fin r) V F)
    (trace : ExactRowTrace F V (Set.range A.row) (Set.range P.row))
    (pivots : Fin r → V) (diagonal : ∀ i, P i (pivots i) ≠ 0)
    (earlier_zero : ∀ i j, j < i → P i (pivots j) = 0) : A.rank = r := by
  calc
    A.rank = P.rank := by
      rw [Matrix.rank_eq_finrank_span_row, Matrix.rank_eq_finrank_span_row,
        exact_row_trace_span trace]
    _ = r := echelon_rows_rank P pivots diagonal earlier_zero

/-- Mathematical evidence for a completed rational fallback. No rank field is assumed. -/
structure ExactEchelonCertificate (A : Matrix I V ℚ) where
  count : ℕ
  rows : Matrix (Fin count) V ℚ
  pivots : Fin count → V
  trace : ExactRowTrace ℚ V (Set.range A.row) (Set.range rows.row)
  diagonal : ∀ i, rows i (pivots i) ≠ 0
  earlier_zero : ∀ i j, j < i → rows i (pivots j) = 0

/-- Only a completed proof trace supplies a numerical answer; absence supplies none. -/
def exactFallbackValue {A : Matrix I V ℚ} (c : Option (ExactEchelonCertificate A)) : Option ℕ :=
  c.map (fun evidence => evidence.count)

theorem exact_fallback_normal_return [Fintype I] [Fintype V] {A : Matrix I V ℚ}
    {c : Option (ExactEchelonCertificate A)} {r : ℕ}
    (returned : exactFallbackValue c = some r) : A.rank = r := by
  cases c with
  | none => simp [exactFallbackValue] at returned
  | some evidence =>
    have heq : evidence.count = r := by simpa [exactFallbackValue] using returned
    exact (exact_trace_rank A evidence.rows evidence.trace evidence.pivots
      evidence.diagonal evidence.earlier_zero).trans heq

theorem exact_fallback_unresolved {A : Matrix I V ℚ} :
    exactFallbackValue (A := A) none = none := rfl

end EconHDFE
