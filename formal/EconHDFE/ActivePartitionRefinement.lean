import EconHDFE.PartitionRefinement

/-!
# Active and reference-coded categorical blocks

Completeness is required only on the realized coarse cell being reduced.
Unobserved labels may be absent. The retained coarse witness and every other
required fine column must remain. Common multipliers act on both sides.
-/
noncomputable section
namespace EconHDFE
open scoped Classical BigOperators
variable {I A B : Type*}

def activeIndicatorSpace (Q : I → B) (active : Finset B) : Submodule ℝ (I → ℝ) :=
  Submodule.span ℝ (indicator Q '' (↑active : Set B))

/-- The full-sum identity needs only coverage of realized levels in this cell. -/
theorem coarse_active_sum (P : I → A) (Q : I → B) (f : B → A)
    (h : ∀ i, P i = f (Q i)) (active : Finset B) (p : A)
    (complete : ∀ i, P i = p → Q i ∈ active) :
    indicator P p = ∑ q ∈ active, if f q = p then indicator Q q else 0 := by
  ext i
  simp only [Finset.sum_apply]
  have heq : (fun q => if f q = p then indicator Q q i else 0) =
      (fun q => if q = Q i then (if f q = p then (1 : ℝ) else 0) else 0) := by
    funext q
    by_cases hq : q = Q i
    · subst q
      simp [indicator]
    · have hqi : Q i ≠ q := Ne.symm hq
      simp [indicator, hq, hqi]
  rw [heq]
  by_cases hp : P i = p
  · have hmem := complete i hp
    simp [indicator, hp, hmem, ← h i]
  · have hn : f (Q i) ≠ p := by simpa only [← h i] using hp
    simp [indicator, hp, hn]

/-- One represented fine column may be dropped while its coarse witness is retained. -/
theorem drop_active_fine (P : I → A) (Q : I → B) (f : B → A)
    (h : ∀ i, P i = f (Q i)) (active : Finset B) (q₀ : B)
    (hq₀ : q₀ ∈ active) (R : Submodule ℝ (I → ℝ))
    (coarse : indicator P (f q₀) ∈ R)
    (complete : ∀ i, P i = f q₀ → Q i ∈ active) :
    R ⊔ activeIndicatorSpace Q active =
      R ⊔ activeIndicatorSpace Q (active.erase q₀) := by
  let S := R ⊔ activeIndicatorSpace Q (active.erase q₀)
  have kept : ∀ q ∈ active.erase q₀, indicator Q q ∈ S := by
    intro q hq
    exact (show activeIndicatorSpace Q (active.erase q₀) ≤ S from le_sup_right)
      (Submodule.subset_span ⟨q, hq, rfl⟩)
  have hs : (∑ q ∈ active.erase q₀, if f q = f q₀ then indicator Q q else 0) ∈ S := by
    apply Submodule.sum_mem
    intro q hq
    by_cases hh : f q = f q₀
    · simpa only [if_pos hh] using kept q hq
    · simp only [if_neg hh]
      exact S.zero_mem
  have pivot : indicator Q q₀ ∈ S := by
    have sumid := coarse_active_sum P Q f h active (f q₀) complete
    have eraseid := Finset.sum_erase_add (s := active)
      (f := fun q => if f q = f q₀ then indicator Q q else 0) hq₀
    simp only [if_pos rfl] at eraseid
    have hp : indicator P (f q₀) ∈ S := (show R ≤ S from le_sup_left) coarse
    have he : indicator Q q₀ = indicator P (f q₀) -
        ∑ q ∈ active.erase q₀, if f q = f q₀ then indicator Q q else 0 := by
      rw [sumid, ← eraseid]
      abel
    rw [he]
    exact S.sub_mem hp hs
  apply le_antisymm
  · apply sup_le le_sup_left
    apply Submodule.span_le.mpr
    rintro _ ⟨q, hq, rfl⟩
    by_cases he : q = q₀
    · simpa only [he] using pivot
    · exact kept q (Finset.mem_erase.mpr ⟨he, hq⟩)
  · apply sup_le le_sup_left
    apply Submodule.span_le.mpr
    rintro _ ⟨q, hq, rfl⟩
    exact (show activeIndicatorSpace Q active ≤ R ⊔ activeIndicatorSpace Q active
      from le_sup_right)
      (Submodule.subset_span ⟨q, (Finset.mem_erase.mp hq).2, rfl⟩)

/-- Multiplication cannot break an equality of column spaces. -/
theorem common_multiplier_space_eq (m : I → ℝ)
    {S T : Submodule ℝ (I → ℝ)} (h : S = T) :
    S.map (multiplier m) = T.map (multiplier m) := congrArg _ h

/-- The coarse witness is multiplied too; no sign or nonvanishing condition on m. -/
theorem drop_active_fine_multiplier (P : I → A) (Q : I → B) (f : B → A)
    (h : ∀ i, P i = f (Q i)) (active : Finset B) (q₀ : B)
    (hq₀ : q₀ ∈ active) (R : Submodule ℝ (I → ℝ))
    (coarse : indicator P (f q₀) ∈ R)
    (complete : ∀ i, P i = f q₀ → Q i ∈ active) (m : I → ℝ) :
    (R ⊔ activeIndicatorSpace Q active).map (multiplier m) =
      (R ⊔ activeIndicatorSpace Q (active.erase q₀)).map (multiplier m) :=
  common_multiplier_space_eq m (drop_active_fine P Q f h active q₀ hq₀ R coarse complete)

/-- Additional unchanged blocks may be retained around an exact reduction. -/
theorem reduction_in_context (R : Submodule ℝ (I → ℝ))
    {S T : Submodule ℝ (I → ℝ)} (h : S = T) : R ⊔ S = R ⊔ T := congrArg _ h

/-- A finite certified reduction remains exact under a common multiplier. -/
theorem finite_reduction_common_multiplier (n : ℕ)
    (spaces : ℕ → Submodule ℝ (I → ℝ))
    (step : ∀ k < n, spaces k = spaces (k + 1)) (m : I → ℝ) :
    (spaces 0).map (multiplier m) = (spaces n).map (multiplier m) :=
  common_multiplier_space_eq m (finite_reduction_preserves_space n spaces step)

end EconHDFE
