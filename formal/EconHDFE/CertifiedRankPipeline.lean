import EconHDFE.RankPipeline
import EconHDFE.ModularRankCertificate
import EconHDFE.ExactRowReduction

/-!
# Composition of structural reductions and exact arithmetic evidence

Every unresolved integer core block needs either a modular hit at a valid
upper bound or a completed rational row-reduction trace. These are mathematical
evidence objects, not trusted outputs from an unverified numerical backend.
-/
noncomputable section
namespace EconHDFE
open scoped Classical BigOperators
variable {I V : Type*} [Fintype I] [Fintype V]

/-- Neither constructor assumes that the requested characteristic-zero rank is correct. -/
inductive IntegerRankEvidence (A : Matrix I V ℤ) (r : ℕ) : Prop where
  | modular (p : ℕ) (prime : p.Prime)
      (upper : (A.map (Int.castRingHom ℚ)).rank ≤ r)
      (hit : (A.map (Int.castRingHom (ZMod p))).rank = r) : IntegerRankEvidence A r
  | rational (certificate : ExactEchelonCertificate (A.map (Int.castRingHom ℚ)))
      (count : certificate.count = r) : IntegerRankEvidence A r

/-- Both accepted evidence routes establish the actual rational rank. -/
theorem integer_rank_evidence_sound {A : Matrix I V ℤ} {r : ℕ}
    (evidence : IntegerRankEvidence A r) : (A.map (Int.castRingHom ℚ)).rank = r := by
  cases evidence with
  | modular p prime upper hit =>
    letI : Fact p.Prime := ⟨prime⟩
    exact modular_rank_exact A upper hit
  | rational certificate count =>
    exact (exact_trace_rank _ certificate.rows certificate.trace certificate.pivots
      certificate.diagonal certificate.earlier_zero).trans count

/-- Scalar extension also certifies the real rank used by FE projection. -/
theorem integer_rank_evidence_real {A : Matrix I V ℤ} {r : ℕ}
    (evidence : IntegerRankEvidence A r) : (A.map (Int.castRingHom ℝ)).rank = r :=
  (integer_rank_charZero ℝ A).trans (integer_rank_evidence_sound evidence)

variable {J G H : Type*} {K : G → Type*}

/-- Original observations, representative tuples, legal peeling and certified core blocks. -/
theorem categorical_certified_rank [Fintype J] [Fintype G]
    [∀ g, Fintype (K g)] [Fintype H]
    {R C : H → Type*} [∀ h, Fintype (R h)] [∀ h, Fintype (C h)]
    (code : (g : G) → I → K g) (rows : J → I)
    (cover : ∀ i, ∃ j, ∀ g, code g i = code g (rows j))
    (t : PeelingTrace (fun g j => code g (rows j)))
    (edgeIndex : PeelingCore t ≃ Sigma R) (levelIndex : Sigma K ≃ Sigma C)
    (separate : ∀ i v, (edgeIndex i).1 ≠ (levelIndex v).1 →
      code v.1 (rows i.val) ≠ v.2)
    (reported : H → ℕ)
    (evidence : ∀ h, IntegerRankEvidence
      ((incidenceMatrix ℤ (coreCode t)).submatrix
        (fun i : R h => edgeIndex.symm ⟨h, i⟩)
        (fun v : C h => levelIndex.symm ⟨h, v⟩)) (reported h)) :
    (incidenceMatrix ℝ code).rank = t.length + ∑ h, reported h := by
  rw [categorical_structural_rank code rows cover t edgeIndex levelIndex separate]
  congr 1
  apply Finset.sum_congr rfl
  intro h _
  have he := integer_rank_evidence_real (evidence h)
  have hmap :
      (((incidenceMatrix ℤ (coreCode t)).submatrix
        (fun i : R h => edgeIndex.symm ⟨h, i⟩)
        (fun v : C h => levelIndex.symm ⟨h, v⟩)).map (Int.castRingHom ℝ)) =
      ((incidenceMatrix ℝ (coreCode t)).submatrix
        (fun i : R h => edgeIndex.symm ⟨h, i⟩)
        (fun v : C h => levelIndex.symm ⟨h, v⟩)) := by
    change ((incidenceMatrix ℤ (coreCode t)).map (Int.castRingHom ℝ)).submatrix _ _ = _
    rw [incidence_int_map]
  rwa [hmap] at he

end EconHDFE
