import EconHDFE.RankBasics
import EconHDFE.PeelingProjection
import EconHDFE.PeelingTermination

/-!
# Exact categorical peeling rank recursion

The rank increment is derived from the already proved incidence-level
reconstruction, not postulated. Each removed observation contributes one
coordinate, even when several singleton levels select the same observation.
These statements concern real matrix rank; cross-field equivalence and the
characteristic-zero elimination program are separate obligations.
-/
noncomputable section
namespace EconHDFE
open scoped Classical BigOperators
variable {I G : Type*} {K : G → Type*} {code : (g : G) → I → K g}

def peelingRows (t : PeelingTrace code) : Fin t.length ⊕ PeelingCore t → I :=
  Sum.elim t.row Subtype.val

theorem peelingRows_bijective (t : PeelingTrace code) : Function.Bijective (peelingRows t) := by
  constructor
  · intro a b hab
    cases a with
    | inl k =>
      cases b with
      | inl j => exact congrArg Sum.inl (t.row_injective hab)
      | inr c => exact False.elim (c.property k hab)
    | inr c =>
      cases b with
      | inl k => exact False.elim (c.property k hab.symm)
      | inr d => exact congrArg Sum.inr (Subtype.ext hab)
  · intro i
    by_cases hi : ∃ k, t.row k = i
    · obtain ⟨k, hk⟩ := hi
      exact ⟨Sum.inl k, hk⟩
    · have hc : i ∈ surviving t := fun k hk => hi ⟨k, hk⟩
      exact ⟨Sum.inr ⟨i, hc⟩, rfl⟩

def peelingRowsEquiv (t : PeelingTrace code) : Fin t.length ⊕ PeelingCore t ≃ I :=
  Equiv.ofBijective (peelingRows t) (peelingRows_bijective t)

/-- Split an attainable fit into peeled coordinates and its attainable core fit. -/
def peelingFitMap (t : PeelingTrace code) :
    feSpace code →ₗ[ℝ] ((Fin t.length → ℝ) × feSpace (coreCode t)) where
  toFun z := (fun k => z.val (t.row k),
    ⟨coreRestriction t z.val, (peeling_feasible_iff t z.val).mp z.property⟩)
  map_add' x y := rfl
  map_smul' a x := rfl

theorem peelingFitMap_bijective (t : PeelingTrace code) :
    Function.Bijective (peelingFitMap t) := by
  constructor
  · intro x y hxy
    apply Subtype.ext
    funext i
    by_cases hi : ∃ k, t.row k = i
    · obtain ⟨k, rfl⟩ := hi
      exact congrFun (congrArg Prod.fst hxy) k
    · have hc : i ∈ surviving t := fun k hk => hi ⟨k, hk⟩
      exact congrArg (fun p : (Fin t.length → ℝ) × feSpace (coreCode t) =>
        p.2.val ⟨i, hc⟩) hxy
  · rintro ⟨a, b⟩
    let z : I → ℝ := fun i => Sum.elim a b.val ((peelingRowsEquiv t).symm i)
    have hrow : ∀ k, z (t.row k) = a k := by
      intro k
      change Sum.elim a b.val
        ((peelingRowsEquiv t).symm ((peelingRowsEquiv t) (Sum.inl k))) = a k
      rw [Equiv.symm_apply_apply]
      rfl
    have hcore : ∀ c : PeelingCore t, z c.val = b.val c := by
      intro c
      change Sum.elim a b.val
        ((peelingRowsEquiv t).symm ((peelingRowsEquiv t) (Sum.inr c))) = b.val c
      rw [Equiv.symm_apply_apply]
      rfl
    have hz : z ∈ feSpace code := by
      apply (peeling_feasible_iff t z).mpr
      have heq : coreRestriction t z = b.val := funext hcore
      rw [heq]
      exact b.property
    refine ⟨⟨z, hz⟩, ?_⟩
    apply Prod.ext
    · exact funext hrow
    · apply Subtype.ext
      exact funext hcore

/-- Rank recursion for any legal trace, including an empty or partial trace. -/
theorem categorical_peeling_rank [Fintype I] (t : PeelingTrace code) :
    Module.finrank ℝ (feSpace code) =
      t.length + Module.finrank ℝ (feSpace (coreCode t)) := by
  let e := LinearEquiv.ofBijective (peelingFitMap t) (peelingFitMap_bijective t)
  simpa only [Module.finrank_prod, Module.finrank_fin_fun] using e.finrank_eq

/-- The same identity for Mathlib's actual matrix rank. -/
theorem categorical_peeling_matrix_rank [Fintype I]
    [Fintype G] [∀ g, Fintype (K g)] (t : PeelingTrace code) :
    (incidenceMatrix ℝ code).rank =
      t.length + (incidenceMatrix ℝ (coreCode t)).rank := by
  rw [incidence_rank_eq_finrank, incidence_rank_eq_finrank]
  exact categorical_peeling_rank t

/-- Corrected lem:peel: one valid additional removal adds exactly one rank unit. -/
theorem categorical_leaf_rank [Fintype I] (t : PeelingTrace code) (i : I) (v : Sigma K)
    (hi : i ∈ surviving t)
    (hv : ∀ j, j ∈ surviving t → (code v.1 j = v.2 ↔ j = i)) :
    Module.finrank ℝ (feSpace (coreCode t)) =
      1 + Module.finrank ℝ (feSpace (coreCode (appendPeeling t i v hi hv))) := by
  have h1 := categorical_peeling_rank t
  have h2 := categorical_peeling_rank (appendPeeling t i v hi hv)
  rw [appendPeeling_length] at h2
  omega

theorem peeling_sample_card [Fintype I] (t : PeelingTrace code) :
    Fintype.card I = t.length + Fintype.card (PeelingCore t) := by
  simpa using (Fintype.card_congr (peelingRowsEquiv t)).symm

/-- FE-only residual-space dimension is preserved; this is not a cluster DoF rule. -/
theorem peeling_residual_dimension [Fintype I] (t : PeelingTrace code) :
    Fintype.card I - Module.finrank ℝ (feSpace code) =
      Fintype.card (PeelingCore t) - Module.finrank ℝ (feSpace (coreCode t)) := by
  rw [peeling_sample_card t, categorical_peeling_rank t]
  omega

/-- Rank peeling may follow deduplication; projection peeling may not. -/
theorem dedup_then_peeling_rank {J : Type*} [Fintype I] [Fintype J]
    [Fintype G] [∀ g, Fintype (K g)]
    (code : (g : G) → I → K g) (rows : J → I)
    (cover : ∀ i, ∃ j, ∀ g, code g i = code g (rows j))
    (t : PeelingTrace (fun g j => code g (rows j))) :
    (incidenceMatrix ℝ code).rank =
      t.length + (incidenceMatrix ℝ (coreCode t)).rank := by
  rw [categorical_rank_dedup code rows cover]
  exact categorical_peeling_matrix_rank t

end EconHDFE
