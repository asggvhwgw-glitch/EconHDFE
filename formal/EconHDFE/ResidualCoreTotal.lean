import EconHDFE.LeastSquaresExistence
import EconHDFE.PeelingProjection
import EconHDFE.PeelingTermination
import EconHDFE.CategoricalRecode

/-!
# Total mathematical residual-core statements

Existence of the core projection is derived from finite positive-weight WLS,
not supplied by an assumed optimizer. These statements concern exact linear
spaces, not numerical convergence or the implementation of peeling.
-/
noncomputable section
namespace EconHDFE
open scoped Classical
variable {I G : Type*} {K : G → Type*} [Fintype I]

/-- Core reconstruction for every input, without an assumed residual or attained fit. -/
theorem categorical_residual_core_total {code : (g : G) → I → K g}
    (t : PeelingTrace code) (w y : I → ℝ) (hw : ∀ i, 0 < w i) :
    wResidual w hw (feSpace code) y =
      peelExtend t (wResidual (fun c : PeelingCore t => w c.val)
        (fun c => hw c.val) (feSpace (coreCode t)) (fun c => y c.val)) :=
  categorical_residual_core t hw (wResidual_spec w hw (feSpace code) y)
    (wResidual_spec _ _ _ _)

/-- A terminal core exists and yields the unique full residual for every y. -/
theorem categorical_terminal_core_projection (code : (g : G) → I → K g)
    (w y : I → ℝ) (hw : ∀ i, 0 < w i) :
    ∃ t : PeelingTrace code, NoSingletons code (surviving t) ∧
      wResidual w hw (feSpace code) y =
        peelExtend t (wResidual (fun c : PeelingCore t => w c.val)
          (fun c => hw c.val) (feSpace (coreCode t)) (fun c => y c.val)) := by
  obtain ⟨t, ht⟩ := terminal_trace_exists code
  exact ⟨t, ht, categorical_residual_core_total t w y hw⟩

/-- One legal trace handles every RHS and every strictly positive weight update. -/
theorem categorical_multi_rhs_total {code : (g : G) → I → K g}
    (t : PeelingTrace code) {M : Type*} (W Y : M → I → ℝ)
    (hw : ∀ m i, 0 < W m i) :
    ∀ m, wResidual (W m) (hw m) (feSpace code) (Y m) =
      peelExtend t (wResidual (fun c : PeelingCore t => W m c.val)
        (fun c => hw m c.val) (feSpace (coreCode t)) (fun c => Y m c.val)) :=
  fun m => categorical_residual_core_total t (W m) (Y m) (hw m)

/-- Redensification preserves the total result if realized level equality is unchanged. -/
theorem categorical_reencoded_core_total {H : G → Type*}
    {code : (g : G) → I → K g} (t : PeelingTrace code)
    (recoded : (g : G) → PeelingCore t → H g)
    (same : ∀ g i j, coreCode t g i = coreCode t g j ↔ recoded g i = recoded g j)
    (w y : I → ℝ) (hw : ∀ i, 0 < w i) :
    wResidual w hw (feSpace code) y =
      peelExtend t (wResidual (fun c : PeelingCore t => w c.val)
        (fun c => hw c.val) (feSpace recoded) (fun c => y c.val)) := by
  rw [categorical_residual_core_total t w y hw, feSpace_recode (coreCode t) recoded same]

end EconHDFE
