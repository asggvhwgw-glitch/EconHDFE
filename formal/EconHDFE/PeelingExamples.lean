import EconHDFE.PeelingProjection
import EconHDFE.PeelingTermination

/-!
# Small kernel-checked witnesses

The cascade has tuples 000, 001, 011, 111, 111. Its first three rows peel in
sequence; the two equal tuples survive as distinct observations. These examples
check the intended meaning of the trace, not replace the general proofs.
-/
noncomputable section
namespace EconHDFE
open scoped Classical

def cascadeCode (g : Fin 3) (i : Fin 5) : Fin 2 :=
  if i.val + g.val < 3 then 0 else 1

def cascadeRow (k : Fin 3) : Fin 5 := ⟨k.val, by omega⟩

def cascadePivot (k : Fin 3) : Sigma (fun _ : Fin 3 => Fin 2) :=
  ⟨⟨2 - k.val, by omega⟩, 0⟩

def cascadeTrace : PeelingTrace cascadeCode where
  length := 3
  row := cascadeRow
  row_injective := by
    intro a b h
    exact Fin.ext (congrArg (fun z : Fin 5 => z.val) h)
  pivot := cascadePivot
  singleton := by decide

theorem cascade_survivors : surviving cascadeTrace = {i : Fin 5 | 3 ≤ i.val} := by
  ext i
  change (∀ k : Fin 3, cascadeRow k ≠ i) ↔ 3 ≤ i.val
  fin_cases i <;> decide

theorem cascade_matrix :
    peelMatrix cascadeTrace = !![1, 1, 1; 0, 1, 1; 0, 0, 1] := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [peelMatrix, cascadeTrace, cascadeRow, feColumn, indicator,
      cascadePivot, cascadeCode] <;> decide

theorem cascade_duplicate_core (g : Fin 3) : cascadeCode g 3 = cascadeCode g 4 := by
  fin_cases g <;> decide

theorem cascade_terminal : NoSingletons cascadeCode (surviving cascadeTrace) := by
  intro i hi v hv
  have hil : 3 ≤ i.val := by simpa only [cascade_survivors, Set.mem_setOf_eq] using hi
  by_cases h3 : i = 3
  · subst i
    refine ⟨4, ?_, by decide, ?_⟩
    · rw [cascade_survivors]
      decide
    · exact (cascade_duplicate_core v.1).symm.trans hv
  · have hi4 : i = 4 := by
      apply Fin.ext
      have hlt := i.isLt
      have hne : i.val ≠ 3 := fun h => h3 (Fin.ext h)
      omega
    subst i
    refine ⟨3, ?_, by decide, ?_⟩
    · rw [cascade_survivors]
      decide
    · exact (cascade_duplicate_core v.1).trans hv

def duplicateCode (_ : Fin 3) (_ : Bool) : Fin 1 := 0

theorem duplicate_pair_leafless : NoSingletons duplicateCode Set.univ := by
  intro i _ v _
  refine ⟨!i, Set.mem_univ _, ?_, ?_⟩
  · cases i <;> decide
  · exact Subsingleton.elim _ _

/-- Two identical tuples cannot be collapsed into a spurious singleton. -/
theorem duplicate_trace_empty (t : PeelingTrace duplicateCode) : t.length = 0 := by
  by_contra h
  have hn : 0 < t.length := Nat.pos_of_ne_zero h
  let k : Fin t.length := ⟨0, hn⟩
  have hs := noSingletons_survive t duplicate_pair_leafless (Set.mem_univ (t.row k))
  exact hs k rfl

end EconHDFE
