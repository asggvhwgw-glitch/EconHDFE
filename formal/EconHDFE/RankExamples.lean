import EconHDFE.RankPipeline
import EconHDFE.ProperConnectivityRank
import EconHDFE.PeelingExamples

/-!
# Semantic witnesses for structural rank

Examples connect actual FE tuples, legal traces and the rank formulas. They are
kernel-checked mathematical witnesses, not numerical rank tests or additional
Python regression cases.
-/
noncomputable section
namespace EconHDFE
open scoped Classical BigOperators

/-- Repeated all-equal tuples have rank one, over every field. -/
theorem duplicate_incidence_rank_one (F : Type*) [Field F] :
    (incidenceMatrix F duplicateCode).rank = 1 := by
  have hc : ProperConnected duplicateCode := by
    intro i j
    exact Relation.ReflTransGen.single ⟨0, fun _ _ => rfl⟩
  have ho : ∀ g, Function.Surjective (duplicateCode g) := by
    intro g k
    exact ⟨false, Subsingleton.elim _ _⟩
  simpa [Fintype.card_sigma] using
    proper_connectivity_rank (F := F) duplicateCode false hc ho 0 (fun _ => 0)

def singletonTupleCode (_ : Fin 3) (_ : Unit) : Fin 1 := 0

def singletonTupleTrace : PeelingTrace singletonTupleCode where
  length := 1
  row := fun _ => ()
  row_injective := fun _ _ _ => Subsingleton.elim _ _
  pivot := fun _ => ⟨0, 0⟩
  singleton := by
    intro k i _
    constructor
    · intro _
      exact Subsingleton.elim _ _
    · intro _
      exact Subsingleton.elim _ _

/-- Three singleton levels on one unique row contribute one, not three, rank units. -/
theorem duplicate_rank_after_dedup : (incidenceMatrix ℝ duplicateCode).rank = 1 := by
  letI : IsEmpty (PeelingCore singletonTupleTrace) :=
    ⟨fun c => c.property 0 (Subsingleton.elim _ _)⟩
  have hcore : (incidenceMatrix ℝ (coreCode singletonTupleTrace)).rank = 0 := by
    apply Nat.eq_zero_of_le_zero
    simpa using (incidenceMatrix ℝ (coreCode singletonTupleTrace)).rank_le_card_height
  have h := dedup_then_peeling_rank duplicateCode (fun _ : Unit => false)
    (fun _ => ⟨(), fun _ => rfl⟩) singletonTupleTrace
  simpa only [hcore, add_zero] using h

/-- The cascade's two surviving observations are one categorical fitted dimension. -/
theorem cascade_core_rank_one :
    (incidenceMatrix ℝ (coreCode cascadeTrace)).rank = 1 := by
  let constant : (g : Fin 3) → PeelingCore cascadeTrace → Fin 1 := fun _ _ => 0
  let root : PeelingCore cascadeTrace := ⟨3, by rw [cascade_survivors]; decide⟩
  have same : ∀ g i j,
      coreCode cascadeTrace g i = coreCode cascadeTrace g j ↔ constant g i = constant g j := by
    intro g i j
    have hi : 3 ≤ i.val.val := by
      simpa only [cascade_survivors, Set.mem_setOf_eq] using i.property
    have hj : 3 ≤ j.val.val := by
      simpa only [cascade_survivors, Set.mem_setOf_eq] using j.property
    have hni : ¬i.val.val + g.val < 3 := by omega
    have hnj : ¬j.val.val + g.val < 3 := by omega
    simp [coreCode, cascadeCode, hni, hnj, constant]
  rw [categorical_recode_rank (coreCode cascadeTrace) constant same]
  have hc : ProperConnected constant := by
    intro i j
    exact Relation.ReflTransGen.single ⟨0, fun _ _ => rfl⟩
  have ho : ∀ g, Function.Surjective (constant g) := by
    intro g k
    exact ⟨root, Subsingleton.elim _ _⟩
  simpa [Fintype.card_sigma] using
    proper_connectivity_rank (F := ℝ) constant root hc ho 0 (fun _ => 0)

/-- Actual rank of 000,001,011,111,111 is four, from peeling and core rank. -/
theorem cascade_incidence_rank_four : (incidenceMatrix ℝ cascadeCode).rank = 4 := by
  have h := categorical_peeling_matrix_rank cascadeTrace
  rw [cascade_core_rank_one] at h
  exact h

/-- One FE-only residual dimension remains; duplicate rows were not discarded in WLS. -/
theorem cascade_residual_dimension_one :
    5 - Module.finrank ℝ (feSpace cascadeCode) = 1 := by
  rw [← incidence_rank_eq_finrank, cascade_incidence_rank_four]

end EconHDFE
