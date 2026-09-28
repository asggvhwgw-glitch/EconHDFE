import Mathlib

/-!
# Partition refinement on a fixed sample

Real-valued column spaces are defined as ranges of the actual categorical
maps and identified with spans of indicator columns. The Python planner is
not assumed verified by these mathematical statements.
-/
namespace EconHDFE
open scoped BigOperators
variable {I A B C : Type*}

def Refines (Q : I → B) (P : I → A) : Prop :=
  ∃ f : B → A, ∀ i, P i = f (Q i)

theorem refines_refl (P : I → A) : Refines P P :=
  ⟨id, fun _ => rfl⟩

theorem refines_trans {P : I → A} {Q : I → B} {T : I → C}
    (hQP : Refines Q P) (hTQ : Refines T Q) : Refines T P := by
  rcases hQP with ⟨f, hf⟩
  rcases hTQ with ⟨g, hg⟩
  exact ⟨f ∘ g, fun i => by rw [hf i, hg i]; rfl⟩

/-- Deleting rows cannot invalidate a true functional dependency. -/
theorem refines_restrict {J : Type*} {P : I → A} {Q : I → B}
    (h : Refines Q P) (rows : J → I) :
    Refines (Q ∘ rows) (P ∘ rows) := by
  rcases h with ⟨f, hf⟩
  exact ⟨f, fun j => hf (rows j)⟩

/-- Sufficient component-closure certificate, allowing dependent level types. -/
theorem interaction_refines {U V : Type*} {KA : U → Type*} {KB : V → Type*}
    (P : I → (u : U) → KA u) (Q : I → (v : V) → KB v)
    (pick : U → V) (maps : (u : U) → KB (pick u) → KA u)
    (h : ∀ i u, P i u = maps u (Q i (pick u))) :
    Refines Q P := by
  refine ⟨fun q u => maps u (q (pick u)), ?_⟩
  intro i
  funext u
  exact h i u

def categoricalMap (P : I → A) : (A → ℝ) →ₗ[ℝ] (I → ℝ) where
  toFun a := fun i => a (P i)
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

def partitionSpace (P : I → A) : Submodule ℝ (I → ℝ) :=
  (categoricalMap P).range

def indicator [DecidableEq A] (P : I → A) (a : A) : I → ℝ :=
  fun i => if P i = a then 1 else 0

theorem indicator_mem [DecidableEq A] (P : I → A) (a : A) :
    indicator P a ∈ partitionSpace P :=
  ⟨fun b => if b = a then 1 else 0, rfl⟩

/-- Bridge to the full-dummy column span; not a surrogate definition. -/
theorem partitionSpace_eq_span [Fintype A] [DecidableEq A] (P : I → A) :
    partitionSpace P = Submodule.span ℝ (Set.range (indicator P)) := by
  apply le_antisymm
  · rintro y ⟨a, rfl⟩
    have expand : categoricalMap P a = ∑ k : A, a k • indicator P k := by
      ext i
      simp [categoricalMap, indicator, Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
    rw [expand]
    exact Submodule.sum_mem _ (fun k _ =>
      Submodule.smul_mem _ _ (Submodule.subset_span ⟨k, rfl⟩))
  · apply Submodule.span_le.mpr
    rintro _ ⟨a, rfl⟩
    exact indicator_mem P a

/-- Manuscript lem:dummy. -/
theorem partitionSpace_le {P : I → A} {Q : I → B}
    (h : Refines Q P) : partitionSpace P ≤ partitionSpace Q := by
  rcases h with ⟨f, hf⟩
  rintro y ⟨a, rfl⟩
  refine ⟨a ∘ f, ?_⟩
  ext i
  simp [categoricalMap, Function.comp_apply, hf i]

theorem equivalent_partition_spaces {P : I → A} {Q : I → B}
    (hQP : Refines Q P) (hPQ : Refines P Q) :
    partitionSpace P = partitionSpace Q :=
  le_antisymm (partitionSpace_le hQP) (partitionSpace_le hPQ)

/-- Remove a coarse FE only while its fine witness remains present. -/
theorem canonicalize_one {P : I → A} {Q : I → B}
    (h : Refines Q P) (S : Submodule ℝ (I → ℝ)) :
    (S ⊔ partitionSpace P) ⊔ partitionSpace Q = S ⊔ partitionSpace Q := by
  rw [sup_assoc, sup_eq_right.mpr (partitionSpace_le h)]

def multiplier (m : I → ℝ) : (I → ℝ) →ₗ[ℝ] (I → ℝ) where
  toFun v := fun i => m i * v i
  map_add' x y := by ext i; exact mul_add _ _ _
  map_smul' a x := by ext i; simp [mul_left_comm]

def slopeSpace (P : I → A) (m : I → ℝ) : Submodule ℝ (I → ℝ) :=
  (partitionSpace P).map (multiplier m)

/-- Manuscript lem:slope. Zero and negative multipliers are allowed. -/
theorem shared_multiplier_le {P : I → A} {Q : I → B}
    (h : Refines Q P) (m : I → ℝ) :
    slopeSpace P m ≤ slopeSpace Q m :=
  Submodule.map_mono (partitionSpace_le h)

/-- Exact sum identity on a fully represented coarse cell. -/
theorem coarse_indicator_sum [Fintype B] [DecidableEq A] [DecidableEq B]
    (P : I → A) (Q : I → B) (f : B → A)
    (h : ∀ i, P i = f (Q i)) (p : A) :
    indicator P p =
      ∑ q : B, if f q = p then indicator Q q else 0 := by
  ext i
  simp only [Finset.sum_apply]
  classical
  calc
    indicator P p i =
        (if f (Q i) = p then (1 : ℝ) else 0) := by simp [indicator, h i]
    _ = ∑ q : B, if q = Q i then (if f q = p then 1 else 0) else 0 := by simp
    _ = _ := by
      apply Finset.sum_congr rfl
      intro q _
      by_cases hq : q = Q i
      · subst q
        simp [indicator]
      · have hqi : Q i ≠ q := Ne.symm hq
        simp [indicator, hq, hqi]

/-- Manuscript prop:basis, full-dummy case, with an explicit retained witness. -/
theorem drop_one_fine [Fintype B] [DecidableEq A] [DecidableEq B]
    (P : I → A) (Q : I → B) (f : B → A)
    (h : ∀ i, P i = f (Q i)) (q₀ : B) :
    partitionSpace P ⊔ partitionSpace Q =
    partitionSpace P ⊔ Submodule.span ℝ
      (indicator Q '' {q : B | q ≠ q₀}) := by
  let S := partitionSpace P ⊔ Submodule.span ℝ
    (indicator Q '' {q : B | q ≠ q₀})
  have kept : ∀ q : B, q ≠ q₀ → indicator Q q ∈ S := by
    intro q hq
    exact (show Submodule.span ℝ (indicator Q '' {q : B | q ≠ q₀}) ≤ S
      from le_sup_right) (Submodule.subset_span ⟨q, hq, rfl⟩)
  have erasedSum :
      (∑ q ∈ Finset.univ.erase q₀,
        if f q = f q₀ then indicator Q q else 0) ∈ S := by
    apply Submodule.sum_mem
    intro q hq
    by_cases hh : f q = f q₀
    · simp only [if_pos hh]
      exact kept q (Finset.mem_erase.mp hq).1
    · simp only [if_neg hh]
      exact S.zero_mem
  have pivot : indicator Q q₀ ∈ S := by
    have sumid := coarse_indicator_sum P Q f h (f q₀)
    have eraseid := Finset.sum_erase_add (s := Finset.univ)
      (f := fun q : B => if f q = f q₀ then indicator Q q else 0)
      (Finset.mem_univ q₀)
    simp only [if_pos rfl] at eraseid
    have hc : indicator P (f q₀) ∈ S :=
      (show partitionSpace P ≤ S from le_sup_left) (indicator_mem P (f q₀))
    have heq : indicator Q q₀ =
        indicator P (f q₀) -
        ∑ q ∈ Finset.univ.erase q₀, if f q = f q₀ then indicator Q q else 0 := by
      rw [sumid, ← eraseid]
      abel
    rw [heq]
    exact S.sub_mem hc erasedSum
  apply le_antisymm
  · apply sup_le le_sup_left
    rw [partitionSpace_eq_span]
    apply Submodule.span_le.mpr
    rintro _ ⟨q, rfl⟩
    by_cases hq : q = q₀
    · simpa [hq] using pivot
    · exact kept q hq
  · apply sup_le le_sup_left
    apply Submodule.span_le.mpr
    rintro _ ⟨q, _, rfl⟩
    exact (show partitionSpace Q ≤ partitionSpace P ⊔ partitionSpace Q
      from le_sup_right) (indicator_mem Q q)

/-- Composition of a finite sequence of previously certified equalities. -/
theorem finite_reduction_preserves_space (n : ℕ)
    (spaces : ℕ → Submodule ℝ (I → ℝ))
    (step : ∀ k < n, spaces k = spaces (k + 1)) :
    spaces 0 = spaces n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    exact (ih (fun k hk => step k (Nat.lt_trans hk (Nat.lt_succ_self n)))).trans
      (step n (Nat.lt_succ_self n))

end EconHDFE
