import EconHDFE.CategoricalDesign

/-!
# Completion of finite peeling traces

A surviving singleton extends a trace by one actual observation. The sample
cardinality bounds its length, so a terminal extension exists. Together with
terminal_core_unique this establishes existence and order independence of the
mathematical core, without making a claim about the production queue program.
-/
noncomputable section
namespace EconHDFE
open scoped Classical
variable {I G : Type*} {K : G → Type*} {code : (g : G) → I → K g}

def emptyPeeling (code : (g : G) → I → K g) : PeelingTrace code where
  length := 0
  row := Fin.elim0
  row_injective := by intro i; exact Fin.elim0 i
  pivot := Fin.elim0
  singleton := by intro k; exact Fin.elim0 k

/-- Extend by a level whose support on the current survivor set is exactly {i}. -/
def appendPeeling (t : PeelingTrace code) (i : I) (v : Sigma K)
    (hi : i ∈ surviving t)
    (hv : ∀ j, j ∈ surviving t → (code v.1 j = v.2 ↔ j = i)) :
    PeelingTrace code where
  length := t.length + 1
  row := Fin.lastCases i t.row
  row_injective := by
    intro a b hab
    induction a using Fin.lastCases with
    | last =>
      induction b using Fin.lastCases with
      | last => rfl
      | cast b =>
        have h : i = t.row b := by simpa using hab
        exact False.elim (hi b h.symm)
    | cast a =>
      induction b using Fin.lastCases with
      | last =>
        have h : t.row a = i := by simpa using hab
        exact False.elim (hi a h)
      | cast b =>
        have h : t.row a = t.row b := by simpa using hab
        exact congrArg Fin.castSucc (t.row_injective h)
  pivot := Fin.lastCases v t.pivot
  singleton := by
    intro k j active
    induction k using Fin.lastCases with
    | last =>
      have hj : j ∈ surviving t := by
        intro l
        have h := active l.castSucc (Fin.castSucc_lt_last l)
        simpa using h
      rw [Fin.lastCases_last]
      simpa using hv j hj
    | cast k =>
      have hj : ∀ l, l < k → t.row l ≠ j := by
        intro l hl
        have h := active l.castSucc (by simpa using hl)
        simpa using h
      rw [Fin.lastCases_castSucc]
      simpa using t.singleton k j hj

@[simp]
theorem appendPeeling_length (t : PeelingTrace code) (i : I) (v : Sigma K)
    (hi : i ∈ surviving t)
    (hv : ∀ j, j ∈ surviving t → (code v.1 j = v.2 ↔ j = i)) :
    (appendPeeling t i v hi hv).length = t.length + 1 := rfl

theorem surviving_append (t : PeelingTrace code) (i : I) (v : Sigma K)
    (hi : i ∈ surviving t)
    (hv : ∀ j, j ∈ surviving t → (code v.1 j = v.2 ↔ j = i)) :
    surviving (appendPeeling t i v hi hv) = {j | j ∈ surviving t ∧ j ≠ i} := by
  ext j
  constructor
  · intro h
    constructor
    · intro k
      have hk := h k.castSucc
      simpa [appendPeeling] using hk
    · have hl := h (Fin.last t.length)
      have hij : i ≠ j := by simpa [appendPeeling] using hl
      exact Ne.symm hij
  · rintro ⟨hcore, hne⟩ k
    induction k using Fin.lastCases with
    | last => simpa [appendPeeling] using Ne.symm hne
    | cast k => simpa [appendPeeling] using hcore k

/-- Every finite legal trace can be continued to a terminal trace. -/
theorem terminal_extension_exists [Fintype I] (t : PeelingTrace code) :
    ∃ u : PeelingTrace code,
      NoSingletons code (surviving u) ∧ surviving u ⊆ surviving t := by
  have finish : ∀ m : ℕ, ∀ s : PeelingTrace code,
      Fintype.card I - s.length = m →
      ∃ u : PeelingTrace code,
        NoSingletons code (surviving u) ∧ surviving u ⊆ surviving s := by
    intro m
    induction m using Nat.strong_induction_on with
    | h m ih =>
      intro s hm
      by_cases hs : NoSingletons code (surviving s)
      · exact ⟨s, hs, Set.Subset.refl _⟩
      · have bad : ∃ i, i ∈ surviving s ∧ ∃ v : Sigma K,
            code v.1 i = v.2 ∧
            ∀ j, j ∈ surviving s → code v.1 j = v.2 → j = i := by
          by_contra hbad
          apply hs
          intro i hi v hvi
          by_contra hnone
          apply hbad
          refine ⟨i, hi, v, hvi, ?_⟩
          intro j hj hjv
          by_contra hji
          exact hnone ⟨j, hj, hji, hjv⟩
        obtain ⟨i, hi, v, hvi, huniq⟩ := bad
        have hv : ∀ j, j ∈ surviving s → (code v.1 j = v.2 ↔ j = i) := by
          intro j hj
          constructor
          · exact huniq j hj
          · intro hji
            simpa [hji] using hvi
        let s' := appendPeeling s i v hi hv
        have bound : s.length + 1 ≤ Fintype.card I := peeling_length_le s'
        have smaller : Fintype.card I - s'.length < m := by
          change Fintype.card I - (s.length + 1) < m
          omega
        obtain ⟨u, hu, hus⟩ := ih (Fintype.card I - s'.length) smaller s' rfl
        refine ⟨u, hu, hus.trans ?_⟩
        intro j hj
        have hjs : j ∈ surviving s ∧ j ≠ i := by
          change j ∈ surviving (appendPeeling s i v hi hv) at hj
          rw [surviving_append] at hj
          exact hj
        exact hjs.1
  exact finish _ t rfl

/-- A terminal categorical core exists on every finite observation set. -/
theorem terminal_trace_exists [Fintype I] (code : (g : G) → I → K g) :
    ∃ t : PeelingTrace code, NoSingletons code (surviving t) := by
  obtain ⟨t, ht, _⟩ := terminal_extension_exists (emptyPeeling code)
  exact ⟨t, ht⟩

end EconHDFE
