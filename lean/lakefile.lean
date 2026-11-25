import Lake
open Lake DSL

package lorentzian_matroid where
  leanOptions := #[
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩
  ]

@[default_target]
lean_lib LorentzianMatroid where
  roots := #[`LorentzianMatroid]
