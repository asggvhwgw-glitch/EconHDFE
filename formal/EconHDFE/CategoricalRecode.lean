import EconHDFE.PeelingProjection

/-!
# Core recoding without changing the fitted space

The condition compares equality of labels on actual rows. It permits arbitrary
relabeling and removal of unused levels, so no bijection on unobserved labels
is required. This is the mathematical redensification step, not a verification
of the Python encoder.
-/
noncomputable section
namespace EconHDFE
open scoped Classical
variable {I G : Type*} {K H : G → Type*}

theorem feSpace_le_of_same_partition
    (code : (g : G) → I → K g) (other : (g : G) → I → H g)
    (h : ∀ g i j, code g i = code g j ↔ other g i = other g j) :
    feSpace code ≤ feSpace other := by
  apply Submodule.span_le.mpr
  rintro _ ⟨⟨g, k⟩, rfl⟩
  by_cases hseen : ∃ i, code g i = k
  · obtain ⟨i, hi⟩ := hseen
    have heq : feColumn code ⟨g, k⟩ = feColumn other ⟨g, other g i⟩ := by
      funext j
      have same : code g j = k ↔ other g j = other g i := by
        rw [← hi]
        exact h g j i
      simp only [feColumn, indicator, same]
    rw [heq]
    exact feColumn_mem other ⟨g, other g i⟩
  · have heq : feColumn code ⟨g, k⟩ = 0 := by
      funext i
      have hi : code g i ≠ k := by
        intro hi
        exact hseen ⟨i, hi⟩
      simp [feColumn, indicator, hi]
    rw [heq]
    exact (feSpace other).zero_mem

/-- Label equality on the realized sample determines the categorical space. -/
theorem feSpace_recode
    (code : (g : G) → I → K g) (other : (g : G) → I → H g)
    (h : ∀ g i j, code g i = code g j ↔ other g i = other g j) :
    feSpace code = feSpace other :=
  le_antisymm (feSpace_le_of_same_partition code other h)
    (feSpace_le_of_same_partition other code (fun g i j => (h g i j).symm))

/-- The residual-core result also holds for compactly recoded core levels. -/
theorem categorical_reencoded_residual_core [Fintype I]
    {code : (g : G) → I → K g} (t : PeelingTrace code)
    (recoded : (g : G) → PeelingCore t → H g)
    (hcode : ∀ g i j, coreCode t g i = coreCode t g j ↔ recoded g i = recoded g j)
    {w y r : I → ℝ} (hw : ∀ i, 0 < w i) {rC : PeelingCore t → ℝ}
    (hf : IsResidual w (feSpace code) y r)
    (hc : IsResidual (fun c => w c.val) (feSpace recoded)
      (fun c => y c.val) rC) : r = peelExtend t rC := by
  have hs := feSpace_recode (coreCode t) recoded hcode
  rw [← hs] at hc
  exact categorical_residual_core t hw hf hc

end EconHDFE
