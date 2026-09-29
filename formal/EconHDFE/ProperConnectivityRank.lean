import EconHDFE.MultipartiteRank

/-!
# Proper edge-connectivity rank certificate

A step changes at most one categorical coordinate. With a nonempty sample and
surjective observed-level codes, connectivity by such steps forces every kernel
coefficient to be constant within each partition. The reduced matrix is then
injective, giving the exact V - (G - 1) rank over any field. Ordinary incidence
connectivity alone is not assumed sufficient.
-/
noncomputable section
namespace EconHDFE
open scoped Classical BigOperators
variable {F I G : Type*} {K : G → Type*}
variable [Field F] [Fintype G] [∀ g, Fintype (K g)]

def OneCoordinateStep (code : (g : G) → I → K g) (i j : I) : Prop :=
  ∃ g : G, ∀ h : G, h ≠ g → code h i = code h j

def ProperConnected (code : (g : G) → I → K g) : Prop :=
  ∀ i j, Relation.ReflTransGen (OneCoordinateStep code) i j

/-- Actual incidence multiplication is the sum of selected partition coefficients. -/
theorem incidence_mulVec (code : (g : G) → I → K g) (a : Sigma K → F) (i : I) :
    (incidenceMatrix F code).mulVec a i = ∑ g, a ⟨g, code g i⟩ := by
  simp [Matrix.mulVec, dotProduct, incidenceMatrix, ← Finset.univ_sigma_univ,
    Finset.sum_sigma, ite_mul]

/-- Subtract zero row equations along an actual one-coordinate step. -/
theorem one_coordinate_kernel (code : (g : G) → I → K g) (a : Sigma K → F)
    (hz : ∀ i, ∑ g, a ⟨g, code g i⟩ = 0) {i j : I}
    (hij : OneCoordinateStep code i j) (g : G) :
    a ⟨g, code g i⟩ = a ⟨g, code g j⟩ := by
  obtain ⟨h, hh⟩ := hij
  by_cases hg : g = h
  · subst g
    have hs : (∑ g ∈ Finset.univ.erase h, a ⟨g, code g i⟩) =
        ∑ g ∈ Finset.univ.erase h, a ⟨g, code g j⟩ := by
      apply Finset.sum_congr rfl
      intro g hg
      rw [hh g (Finset.mem_erase.mp hg).1]
    apply add_left_cancel (a := ∑ g ∈ Finset.univ.erase h, a ⟨g, code g i⟩)
    calc
      (∑ g ∈ Finset.univ.erase h, a ⟨g, code g i⟩) + a ⟨h, code h i⟩ =
          ∑ g, a ⟨g, code g i⟩ :=
        Finset.sum_erase_add (s := Finset.univ) (f := fun g => a ⟨g, code g i⟩)
          (Finset.mem_univ h)
      _ = ∑ g, a ⟨g, code g j⟩ := (hz i).trans (hz j).symm
      _ = (∑ g ∈ Finset.univ.erase h, a ⟨g, code g i⟩) + a ⟨h, code h j⟩ := by
        rw [hs]
        exact (Finset.sum_erase_add (s := Finset.univ) (f := fun g => a ⟨g, code g j⟩)
          (Finset.mem_univ h)).symm
  · rw [hh g hg]

/-- Path induction, not a finite enumeration of example hypergraphs. -/
theorem coordinate_path_kernel (code : (g : G) → I → K g) (a : Sigma K → F)
    (hz : ∀ i, ∑ g, a ⟨g, code g i⟩ = 0) {i j : I}
    (hij : Relation.ReflTransGen (OneCoordinateStep code) i j) (g : G) :
    a ⟨g, code g i⟩ = a ⟨g, code g j⟩ := by
  induction hij with
  | refl => rfl
  | @tail j k _ hjk ih => exact ih.trans (one_coordinate_kernel code a hz hjk g)

/-- Surjectivity excludes unused level columns from the exact formula. -/
theorem proper_kernel_levels (code : (g : G) → I → K g) (root : I)
    (connected : ProperConnected code) (observed : ∀ g, Function.Surjective (code g))
    (a : Sigma K → F) (hz : ∀ i, ∑ g, a ⟨g, code g i⟩ = 0) (v : Sigma K) :
    a v = a ⟨v.1, code v.1 root⟩ := by
  rcases v with ⟨g, k⟩
  obtain ⟨i, hi⟩ := observed g k
  change a ⟨g, k⟩ = a ⟨g, code g root⟩
  rw [← hi]
  exact coordinate_path_kernel code a hz (connected i root) g

/-- Extend retained column coefficients by literal zeros on omitted columns. -/
def columnExtend {V : Type*} (p : V → Prop) (a : {v // p v} → F) : V → F :=
  fun v => if h : p v then a ⟨v, h⟩ else 0

theorem matrix_mulVec_columnExtend {V : Type*} [Fintype V]
    (A : Matrix I V F) (p : V → Prop) (a : {v // p v} → F) :
    A.mulVec (columnExtend p a) = (A.submatrix id Subtype.val).mulVec a := by
  funext i
  simp only [Matrix.mulVec, dotProduct]
  calc
    (∑ v, A i v * columnExtend p a v) =
        ∑ v ∈ Finset.univ.filter p, A i v * columnExtend p a v := by
      symm
      apply Finset.sum_subset (Finset.filter_subset _ _)
      intro v _ hv
      have hn : ¬p v := by simpa using hv
      simp [columnExtend, hn]
    _ = ∑ v : {v // p v}, A i v.val * columnExtend p a v.val :=
      Finset.sum_subtype _ (by simp) _
    _ = _ := by
      apply Finset.sum_congr rfl
      intro v _
      simp [columnExtend, v.property]

/-- Connected row equations identify all coefficients of the reduced matrix. -/
theorem proper_reduced_kernel (code : (g : G) → I → K g) (root : I)
    (connected : ProperConnected code) (observed : ∀ g, Function.Surjective (code g))
    (g₀ : G) (base : (g : G) → K g) (a : RetainedLevels g₀ base → F)
    (ha : (reducedIncidenceMatrix F code g₀ base).mulVec a = 0) : a = 0 := by
  let b : Sigma K → F := columnExtend (keptLevel g₀ base) a
  have hb : (incidenceMatrix F code).mulVec b = 0 := by
    rw [matrix_mulVec_columnExtend]
    exact ha
  have hz : ∀ i, ∑ g, b ⟨g, code g i⟩ = 0 := by
    intro i
    rw [← incidence_mulVec code b i]
    exact congrFun hb i
  have hconst := proper_kernel_levels code root connected observed b hz
  let c : G → F := fun g => b ⟨g, code g root⟩
  have hnonref : ∀ g, g ≠ g₀ → c g = 0 := by
    intro g hg
    have hzero : b ⟨g, base g⟩ = 0 := by simp [b, columnExtend, keptLevel, hg]
    exact (hconst ⟨g, base g⟩).symm.trans hzero
  have hsum : (∑ g, c g) = c g₀ := Fintype.sum_eq_single g₀ hnonref
  have href : c g₀ = 0 := hsum.symm.trans (hz root)
  have hc : ∀ g, c g = 0 := by
    intro g
    by_cases hg : g = g₀
    · simpa only [hg] using href
    · exact hnonref g hg
  funext v
  have hv : b v.val = 0 := (hconst v.val).trans (hc v.val.1)
  simpa [b, columnExtend, v.property] using hv

/-- Corrected prop:proper. The result is proved over any field. -/
theorem proper_connectivity_rank [Fintype I] (code : (g : G) → I → K g)
    (root : I) (connected : ProperConnected code)
    (observed : ∀ g, Function.Surjective (code g))
    (g₀ : G) (base : (g : G) → K g) :
    (incidenceMatrix F code).rank =
      Fintype.card (Sigma K) - (Fintype.card G - 1) := by
  let B := reducedIncidenceMatrix F code g₀ base
  have hinj : Function.Injective B.mulVecLin := by
    intro a b hab
    apply sub_eq_zero.mp
    apply proper_reduced_kernel code root connected observed g₀ base
    change B.mulVecLin (a - b) = 0
    rw [map_sub, hab, sub_self]
  have hrank : B.rank = Fintype.card (RetainedLevels g₀ base) := by
    have hk : LinearMap.ker B.mulVecLin = ⊥ := LinearMap.ker_eq_bot.mpr hinj
    have hdim := LinearMap.finrank_range_add_finrank_ker B.mulVecLin
    rw [hk] at hdim
    simpa using hdim
  calc
    (incidenceMatrix F code).rank = B.rank := multipartite_rank_drop code g₀ base
    _ = Fintype.card (RetainedLevels g₀ base) := hrank
    _ = Fintype.card (Sigma K) - (Fintype.card G - 1) := retained_levels_card g₀ base

end EconHDFE
