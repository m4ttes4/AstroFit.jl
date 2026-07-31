# Kernel fields default to Fixed

Every field of a leaf whose model is `<: AbstractKernel` gets a `Fixed` constraint from `_defaults` instead of the usual `Free`; fitting one requires an explicit `@free`. A **Kernel** is normally a known calibration input, and leaving its fields free crashes ForwardDiff on integer fields (`InexactError: Int(Dual)`) and degrades `params(cm)` to an abstract `Vector{Real}`.

Fixing only `Integer` fields was rejected: Int-vs-Float is a proxy for "structural vs fittable" and misfires on float structural fields (oversampling factor, truncation radius), which would reintroduce the same failure later and harder to diagnose.
