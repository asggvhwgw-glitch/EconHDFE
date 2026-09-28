import EconHDFE.PeelingRank
import EconHDFE.ComponentRank
import EconHDFE.CategoricalRecode

/-! Connected structural preprocessing identities, not a numerical-rank oracle. -/
noncomputable section
namespace EconHDFE
open scoped Classical BigOperators
variable {I J G H : Type*} {K : G → Type*}

/-- Actual rank is unchanged by any recoding preserving realized level equality. -/
theorem categorical_recode_rank {K' : G → Type*}
    [Fintype G] [∀ g, Fintype (K g)] [∀ g, Fintype (K' g)]
    (code : (g : G) → I → K g) (other : (g : G) → I → K' g)
    (same : ∀ g i j, code g i = code g j ↔ other g i = other g j) :
    (incidenceMatrix ℝ code).rank = (incidenceMatrix ℝ other).rank := by
  rw [incidence_rank_eq_finrank, incidence_rank_eq_finrank, feSpace_recode code other same]

/-- One composed result from original observations to separate residual rank blocks. -/
theorem categorical_structural_rank [Fintype I] [Fintype J]
    [Fintype G] [∀ g, Fintype (K g)] [Fintype H]
    {R C : H → Type*} [∀ h, Fintype (R h)] [∀ h, Fintype (C h)]
    (code : (g : G) → I → K g) (rows : J → I)
    (cover : ∀ i, ∃ j, ∀ g, code g i = code g (rows j))
    (t : PeelingTrace (fun g j => code g (rows j)))
    (edgeIndex : PeelingCore t ≃ Sigma R) (levelIndex : Sigma K ≃ Sigma C)
    (separate : ∀ i v, (edgeIndex i).1 ≠ (levelIndex v).1 →
      code v.1 (rows i.val) ≠ v.2) :
    (incidenceMatrix ℝ code).rank = t.length + ∑ h,
      ((incidenceMatrix ℝ (coreCode t)).submatrix
        (fun i : R h => edgeIndex.symm ⟨h, i⟩)
        (fun v : C h => levelIndex.symm ⟨h, v⟩)).rank := by
  rw [dedup_then_peeling_rank code rows cover t]
  congr 1
  apply matrix_rank_components
  intro i v hiv
  simp [incidenceMatrix, coreCode, separate i v hiv]

end EconHDFE
