import EconHDFE.CharZeroRank
import EconHDFE.MultipartiteRank

/-!
# Reliable modular rank acceptance

Nonzero minors supply the finite-field lower bound. Meeting an independently
proved upper bound certifies the characteristic-zero rank. A modular shortfall
is not a deficiency certificate. No numerical modular kernel is trusted here.
-/
noncomputable section
namespace EconHDFE
open scoped Classical
variable {I V : Type*} [Fintype I] [Fintype V]

/-- A nonzero modular minor remains nonzero in every characteristic-zero field. -/
theorem modular_minor_nonzero_in_charZero
    {F : Type*} [Field F] [CharZero F] {p r : ℕ} [Fact p.Prime]
    (A : Matrix I V ℤ) (rows : Fin r → I) (cols : Fin r → V)
    (hmod : ((A.map (Int.castRingHom (ZMod p))).submatrix rows cols).det ≠ 0) :
    ((A.map (Int.castRingHom F)).submatrix rows cols).det ≠ 0 := by
  rw [minor_det_map] at hmod ⊢
  have hInt : (A.submatrix rows cols).det ≠ 0 := by
    intro hz
    exact hmod (by rw [hz, map_zero])
  change (↑((A.submatrix rows cols).det) : F) ≠ 0
  exact_mod_cast hInt

/-- A supplied nonzero modular minor lower-bounds characteristic-zero rank. -/
theorem modular_minor_rank_lower
    {F : Type*} [Field F] [CharZero F] {p r : ℕ} [Fact p.Prime]
    (A : Matrix I V ℤ) (rows : Fin r → I) (cols : Fin r → V)
    (hmod : ((A.map (Int.castRingHom (ZMod p))).submatrix rows cols).det ≠ 0) :
    r ≤ (A.map (Int.castRingHom F)).rank :=
  rank_lower_of_nonzero_minor _ rows cols
    (modular_minor_nonzero_in_charZero A rows cols hmod)

/-- Corrected finite-field lower bound; the needed minor is proved to exist. -/
theorem modular_rank_le_charZero
    {F : Type*} [Field F] [CharZero F] (p : ℕ) [Fact p.Prime]
    (A : Matrix I V ℤ) :
    (A.map (Int.castRingHom (ZMod p))).rank ≤ (A.map (Int.castRingHom F)).rank := by
  obtain ⟨rows, cols, _, _, hd⟩ := matrix_exists_rank_minor (A.map (Int.castRingHom (ZMod p)))
  exact modular_minor_rank_lower A rows cols hd

/-- A minor reaching a proved upper bound certifies exact rank. -/
theorem modular_minor_rank_exact
    {F : Type*} [Field F] [CharZero F] {p upper : ℕ} [Fact p.Prime]
    (A : Matrix I V ℤ) (rows : Fin upper → I) (cols : Fin upper → V)
    (hupper : (A.map (Int.castRingHom F)).rank ≤ upper)
    (hmod : ((A.map (Int.castRingHom (ZMod p))).submatrix rows cols).det ≠ 0) :
    (A.map (Int.castRingHom F)).rank = upper :=
  le_antisymm hupper (modular_minor_rank_lower A rows cols hmod)

/-- Equality of modular rank and a valid upper bound is an exact acceptance rule. -/
theorem modular_rank_exact
    {F : Type*} [Field F] [CharZero F] {p upper : ℕ} [Fact p.Prime]
    (A : Matrix I V ℤ)
    (hupper : (A.map (Int.castRingHom F)).rank ≤ upper)
    (hmeet : (A.map (Int.castRingHom (ZMod p))).rank = upper) :
    (A.map (Int.castRingHom F)).rank = upper := by
  have hl := modular_rank_le_charZero (F := F) p A
  rw [hmeet] at hl
  exact le_antisymm hupper hl

/-- The complete modular lower bound and rational/real equality for the same integer matrix. -/
theorem modular_rational_real_sandwich (p : ℕ) [Fact p.Prime] (A : Matrix I V ℤ) :
    (A.map (Int.castRingHom (ZMod p))).rank ≤ (A.map (Int.castRingHom ℚ)).rank ∧
    (A.map (Int.castRingHom ℚ)).rank = (A.map (Int.castRingHom ℝ)).rank :=
  ⟨modular_rank_le_charZero p A, integer_rational_real_rank A⟩

/-- The lower bound applies to actual categorical FE incidence matrices. -/
theorem categorical_modular_rank_le {G : Type*} {K : G → Type*}
    [Fintype G] [∀ g, Fintype (K g)] (p : ℕ) [Fact p.Prime]
    (code : (g : G) → I → K g) :
    (incidenceMatrix (ZMod p) code).rank ≤ (incidenceMatrix ℝ code).rank := by
  simpa only [incidence_int_map] using
    modular_rank_le_charZero (F := ℝ) p (incidenceMatrix ℤ code)

/-- Compose the proved multipartite upper bound with modular acceptance. -/
theorem categorical_modular_rank_exact {G : Type*} {K : G → Type*}
    [Fintype G] [∀ g, Fintype (K g)] (p : ℕ) [Fact p.Prime]
    (code : (g : G) → I → K g) (g₀ : G) (base : (g : G) → K g)
    (hmeet : (incidenceMatrix (ZMod p) code).rank =
      min (Fintype.card I) (Fintype.card (Sigma K) - (Fintype.card G - 1))) :
    (incidenceMatrix ℝ code).rank =
      min (Fintype.card I) (Fintype.card (Sigma K) - (Fintype.card G - 1)) := by
  apply le_antisymm (multipartite_rank_upper code g₀ base)
  rw [← hmeet]
  exact categorical_modular_rank_le p code

end EconHDFE
