import EconHDFE.RankMinors

/-! Characteristic-zero invariance for actual rational and integer matrices. -/
noncomputable section
namespace EconHDFE
open scoped Classical
variable {I V G : Type*} {K : G → Type*} [Fintype I] [Fintype V]

/-- Extending the scalar field from rationals to reals preserves matrix rank. -/
theorem rational_real_rank (A : Matrix I V ℚ) :
    (A.map (algebraMap ℚ ℝ)).rank = A.rank :=
  matrix_rank_map_field (algebraMap ℚ ℝ) A

/-- Integer entries have the same rank in any characteristic-zero field as in Q. -/
theorem integer_rank_charZero (F : Type*) [Field F] [CharZero F]
    (A : Matrix I V ℤ) :
    (A.map (Int.castRingHom F)).rank = (A.map (Int.castRingHom ℚ)).rank := by
  have heq : (A.map (Int.castRingHom ℚ)).map (algebraMap ℚ F) =
      A.map (Int.castRingHom F) := by
    ext i j
    change (((A i j : ℤ) : ℚ) : F) = ((A i j : ℤ) : F)
    simp
  rw [← heq]
  exact matrix_rank_map_field (algebraMap ℚ F) _

/-- Corrected characteristic-zero rank lemma, explicitly for integer matrices. -/
theorem integer_rational_real_rank (A : Matrix I V ℤ) :
    (A.map (Int.castRingHom ℚ)).rank = (A.map (Int.castRingHom ℝ)).rank :=
  (integer_rank_charZero ℝ A).symm

/-- The scalar cast of actual categorical indicators is the same incidence matrix. -/
theorem incidence_int_map (F : Type*) [Ring F] (code : (g : G) → I → K g) :
    (incidenceMatrix ℤ code).map (Int.castRingHom F) = incidenceMatrix F code := by
  ext i v
  by_cases h : code v.1 i = v.2 <;> simp [incidenceMatrix, h]

/-- The real projection library and rational exact-rank target agree. -/
theorem categorical_rational_real_rank [Fintype G] [∀ g, Fintype (K g)]
    (code : (g : G) → I → K g) :
    (incidenceMatrix ℚ code).rank = (incidenceMatrix ℝ code).rank := by
  simpa only [incidence_int_map] using integer_rational_real_rank (incidenceMatrix ℤ code)

end EconHDFE
