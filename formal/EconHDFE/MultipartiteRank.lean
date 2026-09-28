import EconHDFE.RankBasics

/-!
# Multipartite column removal and a proved rank upper bound

Keep every column of one reference partition and remove one specified column
from every other complete categorical block. The actual column span is
preserved over every field. The exact count of retained columns yields the
V - (G - 1) upper bound; it is an upper bound, not an exact-rank assertion.
-/
noncomputable section
namespace EconHDFE
open scoped Classical BigOperators
variable {F I G : Type*} {K : G → Type*}
variable [Field F] [Fintype G] [∀ g, Fintype (K g)]

/-- Every complete categorical block has the same all-ones column sum. -/
omit [Fintype G] in
theorem incidence_partition_sum (code : (g : G) → I → K g) (g : G) :
    (∑ k : K g, (incidenceMatrix F code).col ⟨g, k⟩) = (fun _ : I => (1 : F)) := by
  ext i
  simp [Matrix.col, incidenceMatrix, Finset.sum_apply]

def keptLevel (g₀ : G) (base : (g : G) → K g) (v : Sigma K) : Prop :=
  v.1 = g₀ ∨ v.2 ≠ base v.1

abbrev RetainedLevels (g₀ : G) (base : (g : G) → K g) :=
  {v : Sigma K // keptLevel g₀ base v}

def reducedIncidenceMatrix (F : Type*) [Zero F] [One F]
    (code : (g : G) → I → K g) (g₀ : G) (base : (g : G) → K g) :
    Matrix I (RetainedLevels g₀ base) F :=
  (incidenceMatrix F code).submatrix id Subtype.val

/-- Corrected manuscript lem:blockdrop: each omitted column is reconstructed. -/
theorem multipartite_column_span (code : (g : G) → I → K g)
    (g₀ : G) (base : (g : G) → K g) :
    Submodule.span F (Set.range (incidenceMatrix F code).col) =
      Submodule.span F (Set.range (reducedIncidenceMatrix F code g₀ base).col) := by
  let S := Submodule.span F (Set.range (reducedIncidenceMatrix F code g₀ base).col)
  have keep : ∀ v : Sigma K, keptLevel g₀ base v → (incidenceMatrix F code).col v ∈ S := by
    intro v hv
    exact Submodule.subset_span ⟨⟨v, hv⟩, rfl⟩
  have hone : (fun _ : I => (1 : F)) ∈ S := by
    rw [← incidence_partition_sum (F := F) code g₀]
    apply Submodule.sum_mem
    intro k _
    exact keep ⟨g₀, k⟩ (Or.inl rfl)
  have allcols : ∀ v : Sigma K, (incidenceMatrix F code).col v ∈ S := by
    rintro ⟨g, k⟩
    by_cases hkeep : keptLevel g₀ base ⟨g, k⟩
    · exact keep ⟨g, k⟩ hkeep
    · have hk : k = base g := by
        by_contra hn
        exact hkeep (Or.inr hn)
      have hsum : (∑ a ∈ Finset.univ.erase k,
          (incidenceMatrix F code).col ⟨g, a⟩) ∈ S := by
        apply Submodule.sum_mem
        intro a ha
        apply keep ⟨g, a⟩
        apply Or.inr
        intro hab
        exact (Finset.mem_erase.mp ha).1 (hab.trans hk.symm)
      have hid : (incidenceMatrix F code).col ⟨g, k⟩ =
          (fun _ : I => (1 : F)) -
            ∑ a ∈ Finset.univ.erase k, (incidenceMatrix F code).col ⟨g, a⟩ := by
        have he := Finset.sum_erase_add (s := Finset.univ)
          (f := fun a : K g => (incidenceMatrix F code).col ⟨g, a⟩)
          (Finset.mem_univ k)
        rw [← incidence_partition_sum (F := F) code g, ← he]
        abel
      rw [hid]
      exact S.sub_mem hone hsum
  apply le_antisymm
  · apply Submodule.span_le.mpr
    rintro _ ⟨v, rfl⟩
    exact allcols v
  · apply Submodule.span_le.mpr
    rintro _ ⟨v, rfl⟩
    exact Submodule.subset_span ⟨v.val, rfl⟩

theorem multipartite_rank_drop (code : (g : G) → I → K g)
    (g₀ : G) (base : (g : G) → K g) :
    (incidenceMatrix F code).rank = (reducedIncidenceMatrix F code g₀ base).rank := by
  rw [Matrix.rank_eq_finrank_span_cols, Matrix.rank_eq_finrank_span_cols,
    multipartite_column_span code g₀ base]

/-- Exactly one omitted column for each non-reference partition. -/
def removedLevelsEquiv (g₀ : G) (base : (g : G) → K g) :
    {v : Sigma K // ¬keptLevel g₀ base v} ≃ {g : G // g ≠ g₀} where
  toFun v := ⟨v.val.1, fun h => v.property (Or.inl h)⟩
  invFun g := ⟨⟨g.val, base g.val⟩, by simp [keptLevel, g.property]⟩
  left_inv v := by
    apply Subtype.ext
    rcases v with ⟨⟨g, k⟩, hv⟩
    have hk : k = base g := by
      by_contra hn
      exact hv (Or.inr hn)
    change (⟨g, base g⟩ : Sigma K) = ⟨g, k⟩
    rw [hk]
  right_inv g := by
    apply Subtype.ext
    rfl

theorem nonreference_card (g₀ : G) :
    Fintype.card {g : G // g ≠ g₀} = Fintype.card G - 1 := by
  have hs : Fintype.card {g : G // g = g₀} = 1 := by
    rw [Fintype.card_subtype]
    simp
  simpa only [hs] using Fintype.card_subtype_compl (fun g : G => g = g₀)

theorem retained_levels_card (g₀ : G) (base : (g : G) → K g) :
    Fintype.card (RetainedLevels g₀ base) =
      Fintype.card (Sigma K) - (Fintype.card G - 1) := by
  have hs := Fintype.card_congr (Equiv.sumCompl (keptLevel g₀ base))
  rw [Fintype.card_sum] at hs
  have hr := Fintype.card_congr (removedLevelsEquiv g₀ base)
  rw [nonreference_card] at hr
  change Fintype.card {v : Sigma K // keptLevel g₀ base v} = _
  omega

/-- Corrected prop:upper, with a supplied reference partition and complete blocks. -/
theorem multipartite_rank_upper [Fintype I] (code : (g : G) → I → K g)
    (g₀ : G) (base : (g : G) → K g) :
    (incidenceMatrix F code).rank ≤
      min (Fintype.card I) (Fintype.card (Sigma K) - (Fintype.card G - 1)) := by
  apply le_min (Matrix.rank_le_card_height _)
  rw [multipartite_rank_drop code g₀ base]
  have h := (reducedIncidenceMatrix F code g₀ base).rank_le_card_width
  rwa [retained_levels_card] at h

end EconHDFE
