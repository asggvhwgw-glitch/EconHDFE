import EconHDFE.ModularRankCertificate

/-! Kernel-checked boundary witnesses for modular acceptance. -/
noncomputable section
namespace EconHDFE
open scoped Classical

/-- Every selected prime has an integer matrix on which that prime loses rank. -/
def badPrimeMatrix (p : ℕ) : Matrix (Fin 2) (Fin 2) ℤ := !![(p : ℤ), 0; 0, 1]

theorem bad_prime_rational_rank (p : ℕ) [Fact p.Prime] :
    ((badPrimeMatrix p).map (Int.castRingHom ℚ)).rank = 2 := by
  have hp : (p : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr (Fact.out : p.Prime).ne_zero
  have hd : ((badPrimeMatrix p).map (Int.castRingHom ℚ)).det ≠ 0 := by
    simpa [badPrimeMatrix, Matrix.det_fin_two] using hp
  have hl := rank_lower_of_nonzero_minor
    ((badPrimeMatrix p).map (Int.castRingHom ℚ)) id id (by simpa using hd)
  have hu : ((badPrimeMatrix p).map (Int.castRingHom ℚ)).rank ≤ 2 := by
    simpa only [Fintype.card_fin] using
      ((badPrimeMatrix p).map (Int.castRingHom ℚ)).rank_le_card_width
  exact le_antisymm hu hl

theorem bad_prime_modular_rank (p : ℕ) [Fact p.Prime] :
    ((badPrimeMatrix p).map (Int.castRingHom (ZMod p))).rank = 1 := by
  let M := (badPrimeMatrix p).map (Int.castRingHom (ZMod p))
  have hz : M.det = 0 := by simp [M, badPrimeMatrix, Matrix.det_fin_two]
  have hl : 1 ≤ M.rank := by
    apply rank_lower_of_nonzero_minor M (fun _ : Fin 1 => 1) (fun _ : Fin 1 => 1)
    norm_num [M, badPrimeMatrix, Matrix.det_unique]
  have hu : M.rank ≤ 2 := by simpa only [Fintype.card_fin] using M.rank_le_card_width
  have hn : M.rank ≠ 2 := by
    intro h
    exact square_det_nonzero_of_rank M (by simpa only [Fintype.card_fin] using h) hz
  change M.rank = 1
  omega

/-- Shortfall at an arbitrarily large prime does not prove characteristic-zero deficiency. -/
theorem every_prime_has_rank_shortfall (p : ℕ) [Fact p.Prime] :
    ((badPrimeMatrix p).map (Int.castRingHom (ZMod p))).rank <
      ((badPrimeMatrix p).map (Int.castRingHom ℚ)).rank := by
  rw [bad_prime_modular_rank, bad_prime_rational_rank]
  decide

/-- The same matrix's nonzero minor modulo three certifies its rational rank. -/
theorem good_prime_minor_accepts :
    ((badPrimeMatrix 2).map (Int.castRingHom ℚ)).rank = 2 := by
  letI : Fact (Nat.Prime 3) := ⟨by decide⟩
  apply modular_minor_rank_exact (p := 3) (badPrimeMatrix 2) id id
  · simpa only [Fintype.card_fin] using
      ((badPrimeMatrix 2).map (Int.castRingHom ℚ)).rank_le_card_width
  · norm_num [badPrimeMatrix, Matrix.det_fin_two] <;> decide

/-- A rank-zero witness has the usual nonzero empty determinant, not a fake pivot. -/
theorem empty_minor_det_one {I V : Type*} (A : Matrix I V ℚ) :
    (A.submatrix (Fin.elim0 : Fin 0 → I) (Fin.elim0 : Fin 0 → V)).det = 1 := by
  simp

end EconHDFE
