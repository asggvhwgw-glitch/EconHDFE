import EconHDFE.PartitionRefinement
import EconHDFE.WeightedLeastSquares

/-!
# Residual-core reduction in block form

This proves the corrected appendix's block-algebra result. Translating the
production hypergraph queue into this block form remains a separate proof
obligation, not an implicit claim of this module.
-/
namespace EconHDFE
open scoped BigOperators
variable {L C K : Type*}

def blockMap (T : (L → ℝ) →ₗ[ℝ] (L → ℝ))
    (B : (K → ℝ) →ₗ[ℝ] (L → ℝ))
    (D : (K → ℝ) →ₗ[ℝ] (C → ℝ)) :
    ((L → ℝ) × (K → ℝ)) →ₗ[ℝ] ((L ⊕ C) → ℝ) where
  toFun a := Sum.elim (T a.1 + B a.2) (D a.2)
  map_add' a b := by
    ext i
    cases i <;> simp [map_add, add_assoc, add_left_comm, add_comm]
  map_smul' a x := by
    ext i
    cases i <;> simp [map_smul, smul_add]

def liftCore (S : Submodule ℝ (C → ℝ)) : Submodule ℝ ((L ⊕ C) → ℝ) :=
  S.comap (categoricalMap Sum.inr)

/-- Constructive feasibility; core coefficients need not be unique. -/
theorem block_range_eq_liftCore
    (T : (L → ℝ) →ₗ[ℝ] (L → ℝ))
    (B : (K → ℝ) →ₗ[ℝ] (L → ℝ))
    (D : (K → ℝ) →ₗ[ℝ] (C → ℝ))
    (hT : Function.Surjective T) :
    LinearMap.range (blockMap T B D) = liftCore (LinearMap.range D) := by
  apply le_antisymm
  · rintro z ⟨⟨a, b⟩, rfl⟩
    exact ⟨b, rfl⟩
  · intro z hz
    rcases hz with ⟨b, hb⟩
    obtain ⟨a, ha⟩ := hT ((fun l => z (Sum.inl l)) - B b)
    refine ⟨(a, b), ?_⟩
    ext i
    cases i with
    | inl l =>
      change T a l + B b l = z (Sum.inl l)
      have hh := congrFun ha l
      change T a l = z (Sum.inl l) - B b l at hh
      linarith
    | inr c =>
      exact congrFun hb c

def zeroExtend (r : C → ℝ) : (L ⊕ C) → ℝ := Sum.elim 0 r

theorem normal_transport [Fintype L] [Fintype C]
    (w : (L ⊕ C) → ℝ) (r : C → ℝ) (z : (L ⊕ C) → ℝ) :
    wInner w (zeroExtend r) z =
    wInner (fun c => w (Sum.inr c)) r (fun c => z (Sum.inr c)) := by
  simp [wInner, zeroExtend, Fintype.sum_sum_type]

/-- Feasibility and normal equations lift together. -/
theorem residual_lifts [Fintype L] [Fintype C]
    {w y : (L ⊕ C) → ℝ} {S : Submodule ℝ (C → ℝ)} {r : C → ℝ}
    (hr : IsResidual (fun c => w (Sum.inr c)) S
      (fun c => y (Sum.inr c)) r) :
    IsResidual w (liftCore S) y (zeroExtend r) := by
  constructor
  · exact hr.1
  · intro z hz
    rw [normal_transport]
    exact hr.2 _ hz

theorem block_residual_reconstruction [Fintype L] [Fintype C]
    {w y rFull : (L ⊕ C) → ℝ} (hw : ∀ i, 0 < w i)
    (T : (L → ℝ) →ₗ[ℝ] (L → ℝ))
    (B : (K → ℝ) →ₗ[ℝ] (L → ℝ))
    (D : (K → ℝ) →ₗ[ℝ] (C → ℝ))
    (hT : Function.Surjective T) {rCore : C → ℝ}
    (hf : IsResidual w (LinearMap.range (blockMap T B D)) y rFull)
    (hc : IsResidual (fun c => w (Sum.inr c)) (LinearMap.range D)
      (fun c => y (Sum.inr c)) rCore) :
    rFull = zeroExtend rCore := by
  rw [block_range_eq_liftCore T B D hT] at hf
  exact residual_unique hw hf (residual_lifts hc)

/-- Multiple right-hand sides share the same block identity. -/
theorem block_multi_rhs [Fintype L] [Fintype C] {M : Type*}
    {w : (L ⊕ C) → ℝ} (hw : ∀ i, 0 < w i)
    (T : (L → ℝ) →ₗ[ℝ] (L → ℝ))
    (B : (K → ℝ) →ₗ[ℝ] (L → ℝ))
    (D : (K → ℝ) →ₗ[ℝ] (C → ℝ))
    (hT : Function.Surjective T)
    (Y R : M → (L ⊕ C) → ℝ) (RC : M → C → ℝ)
    (hf : ∀ m, IsResidual w (LinearMap.range (blockMap T B D)) (Y m) (R m))
    (hc : ∀ m, IsResidual (fun c => w (Sum.inr c)) (LinearMap.range D)
      (fun c => Y m (Sum.inr c)) (RC m)) :
    ∀ m, R m = zeroExtend (RC m) := by
  intro m
  exact block_residual_reconstruction hw T B D hT (hf m) (hc m)

end EconHDFE
