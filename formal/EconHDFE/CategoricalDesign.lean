import EconHDFE.PartitionRefinement

/-!
# Categorical incidence and legal peeling traces

Rows are observation identities, not distinct FE tuples. A legal trace selects
an active row and a level incident to exactly that row in the remaining sample.
No matrix-invertibility or residual conclusion is part of the trace definition.
-/
noncomputable section
namespace EconHDFE
open scoped BigOperators Classical
variable {I G : Type*} {K : G → Type*}

/-- A column of the actual full-dummy design. -/
def feColumn (code : (g : G) → I → K g) (v : Sigma K) : I → ℝ :=
  indicator (code v.1) v.2

/-- Algebraic column space; repeated observations remain separate coordinates. -/
def feSpace (code : (g : G) → I → K g) : Submodule ℝ (I → ℝ) :=
  Submodule.span ℝ (Set.range (feColumn code))

theorem feColumn_mem (code : (g : G) → I → K g) (v : Sigma K) :
    feColumn code v ∈ feSpace code :=
  Submodule.subset_span ⟨v, rfl⟩

/-- Actual restriction of the full-dummy design, with original level labels. -/
theorem feSpace_restrict {J : Type*} (code : (g : G) → I → K g) (rows : J → I) :
    (feSpace code).map (categoricalMap rows) =
      feSpace (fun g j => code g (rows j)) := by
  unfold feSpace
  rw [Submodule.map_span]
  congr 1
  ext y
  constructor
  · rintro ⟨x, ⟨v, rfl⟩, rfl⟩
    exact ⟨v, rfl⟩
  · rintro ⟨v, rfl⟩
    exact ⟨feColumn code v, ⟨v, rfl⟩, rfl⟩

structure PeelingTrace (code : (g : G) → I → K g) where
  length : ℕ
  row : Fin length → I
  row_injective : Function.Injective row
  pivot : Fin length → Sigma K
  singleton : ∀ k i, (∀ j, j < k → row j ≠ i) →
    (code (pivot k).1 i = (pivot k).2 ↔ i = row k)

variable {code : (g : G) → I → K g}

def surviving (t : PeelingTrace code) : Set I := {i | ∀ k, t.row k ≠ i}

abbrev PeelingCore (t : PeelingTrace code) := {i : I // i ∈ surviving t}

def coreCode (t : PeelingTrace code) : (g : G) → PeelingCore t → K g :=
  fun g i => code g i.val

def coreRestriction (t : PeelingTrace code) :
    (I → ℝ) →ₗ[ℝ] (PeelingCore t → ℝ) := categoricalMap Subtype.val

/-- Incidence of the selected level with its own row is derived from legality. -/
theorem peeling_pivot_incident (t : PeelingTrace code) (k : Fin t.length) :
    code (t.pivot k).1 (t.row k) = (t.pivot k).2 := by
  apply (t.singleton k (t.row k) ?_).mpr rfl
  intro j hj heq
  exact (ne_of_lt hj) (t.row_injective heq)

theorem peeling_column_diagonal (t : PeelingTrace code) (k : Fin t.length) :
    feColumn code (t.pivot k) (t.row k) = 1 := by
  simp [feColumn, indicator, peeling_pivot_incident t k]

/-- Earlier pivot levels vanish on every later peeled row. -/
theorem peeling_column_later_zero (t : PeelingTrace code)
    {i j : Fin t.length} (hji : j < i) :
    feColumn code (t.pivot j) (t.row i) = 0 := by
  have active : ∀ k, k < j → t.row k ≠ t.row i := by
    intro k hkj heq
    exact (ne_of_lt (lt_trans hkj hji)) (t.row_injective heq)
  have absent : code (t.pivot j).1 (t.row i) ≠ (t.pivot j).2 := by
    intro h
    have heq := (t.singleton j (t.row i) active).mp h
    exact (ne_of_gt hji) (t.row_injective heq)
  simp [feColumn, indicator, absent]

/-- Pivot columns vanish on all surviving observations. -/
theorem peeling_column_core_zero (t : PeelingTrace code)
    (c : PeelingCore t) (k : Fin t.length) :
    feColumn code (t.pivot k) c.val = 0 := by
  have active : ∀ j, j < k → t.row j ≠ c.val := fun j _ => c.property j
  have absent : code (t.pivot k).1 c.val ≠ (t.pivot k).2 := by
    intro h
    exact c.property k ((t.singleton k c.val active).mp h).symm
  simp [feColumn, indicator, absent]

/-- A legal trace cannot remove more rows than the original finite sample. -/
theorem peeling_length_le [Fintype I] (t : PeelingTrace code) :
    t.length ≤ Fintype.card I := by
  simpa using Fintype.card_le_of_injective t.row t.row_injective

/-- Every level that appears in S appears on at least two distinct rows of S. -/
def NoSingletons (code : (g : G) → I → K g) (S : Set I) : Prop :=
  ∀ i, i ∈ S → ∀ v : Sigma K, code v.1 i = v.2 →
    ∃ j, j ∈ S ∧ j ≠ i ∧ code v.1 j = v.2

/-- A leafless subset cannot lose a row in any legal peeling order. -/
theorem noSingletons_survive (t : PeelingTrace code) {S : Set I}
    (hS : NoSingletons code S) : S ⊆ surviving t := by
  have missing : ∀ m : ℕ, ∀ k : Fin t.length, k.val < m → t.row k ∉ S := by
    intro m
    induction m with
    | zero =>
      intro k hk
      omega
    | succ m ih =>
      intro k hk hmem
      by_cases hkm : k.val < m
      · exact ih k hkm hmem
      · have hke : k.val = m := by omega
        obtain ⟨j, hj, hne, hcode⟩ :=
          hS (t.row k) hmem (t.pivot k) (peeling_pivot_incident t k)
        have active : ∀ l, l < k → t.row l ≠ j := by
          intro l hl heq
          have hlm : l.val < m := by
            have hlk : l.val < k.val := hl
            omega
          exact ih l hlm (heq.symm ▸ hj)
        exact hne ((t.singleton k j active).mp hcode)
  intro i hi k heq
  exact missing t.length k k.isLt (heq.symm ▸ hi)

/-- Complete legal traces have the same surviving set, regardless of order. -/
theorem terminal_core_unique (t u : PeelingTrace code)
    (ht : NoSingletons code (surviving t))
    (hu : NoSingletons code (surviving u)) : surviving t = surviving u :=
  le_antisymm (noSingletons_survive u ht) (noSingletons_survive t hu)

end EconHDFE
