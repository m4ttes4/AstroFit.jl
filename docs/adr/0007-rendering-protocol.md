# One evaluator for pointwise and array-native models

## Decision

Keep scalar `render` as the default primitive and concrete array `render` methods
as the Domainwise primitive. `AbstractKernel` declares Domainwise automatically;
ordinary array-native models opt in with `evalstyle(::Type{<:MyModel})`.
Export both `Pointwise` and `Domainwise` so authors can declare the capability.

`_eval` enters a single `_evaluate` traversal shared by allocating rendering,
generic in-place rendering, and general objective reduction. Named leaves and
compiled models unwrap before evaluation. Pointwise subtrees retain their fused
scalar broadcast; domainwise arithmetic combines lazy child expressions.
The old `_arender` traversal is removed.

A pipe passes its left expression directly to internal evaluation of its right
child. A domainwise consumer materializes its input; a pointwise consumer keeps
it lazy. An internal constant flag distinguishes external coordinates from
upstream values. Only the former can invoke the lone-matrix index-grid shorthand.
This preserves `render(model, image)` while preventing intensities produced by
a pipe from silently becoming coordinate indices.

The public generic Domainwise leaf fallback throws for unsupported signatures.
Structural nodes have an explicit public entry point. This lets internal leaf
evaluation call concrete user array methods without a fallback recursion loop.
No new array-render function or registration mechanism is required.

## Contracts and compatibility

- Scalar formulas and existing kernel array methods keep their authoring form.
- Ordinary array-native models receive root coordinates or upstream pipe values;
  defining an array method alone does not change their default Pointwise style.
- Kernel evaluation validates its output axes immediately, before downstream
  broadcasting. A direct user array-method call cannot be intercepted by the
  framework; authors must also honor the contract in that method.
- `render!` destinations must have exactly the prediction's axes. Code relying
  on implicit broadcast expansion must now request that expansion explicitly.
- Mixed scalar/array coordinates use the same broadcasting rules through both
  generic entry points. Unsupported scalar formulas terminate with MethodError;
  unsupported domainwise signatures terminate with an informative ArgumentError.
- A lone external matrix remains a template for pointwise models and array data
  for domainwise leaves. Use explicit broadcast when applying a scalar 1D formula
  to matrix entries. Voigt1D's optimized method now follows this same rule.
- Buffer element types must hold the prediction, including dual numbers.
- Coordinate validation establishes broadcast axes, not physical grid uniformity.

## Performance choices

Retain the optimized Voigt1D and 2D in-place formulas, including their parameter
hoisting. Also retain the specialized 1D pointwise chi2 loop. General pointwise
and domainwise residual reductions share one implementation with final prediction
validation. `check_data` remains independent of model evaluation style.

The initial review on Julia 1.13.0 / Apple M4 measured generic scalar broadcasts
at roughly 2.9 times the specialized Gaussian2D time, 2.2 times Sersic2D, and
12 times Voigt1D. Removing those methods would be a material regression.
Prepared-expression hooks and reusable kernel workspaces are deferred until a
separate workload demonstrates their benefit.

Regression coverage lives in `test/render_protocol_tests.jl`. Benchmarks in
`benchmark/mixed.jl` cover bare consumer costs indirectly through compiled
pointwise, mixed, chained, and transformed trees, objectives, gradients, and
expensive profiles both alone and in sums. Existing handwritten comparisons
remain unchanged.

## Validation

The full suite passed 447 assertions, including 102 new rendering assertions.
The new mixed and profile benchmark groups were also executed. A warmed
before/after comparison against the original source, in the same Julia 1.13.0
process on an Apple M4 with one thread, produced these medians (up to 1,000
samples, 0.3 seconds per trial, one evaluation per sample):

| In-place workload | Before | After | Bytes before → after |
|---|---:|---:|---:|
| Pointwise sum, 512 points | 1.25 µs | 1.25 µs | 0 → 0 |
| Source → GaussianPSF, 512 points | 4.00 µs | 3.88 µs | 8,480 → 8,480 |
| Source → GaussianPSF → Linear1D | 3.92 µs | 4.08 µs | 12,640 → 8,480 |
| Voigt1D, 512 points | 1.21 µs | 1.17 µs | 0 → 0 |
| Gaussian2D, 256×256 grid | 185.4 µs | 185.6 µs | 0 → 0 |
| Sersic2D, 256×256 grid | 352.1 µs | 352.4 µs | 0 → 0 |

These are local measurements, not timing guarantees. The post-kernel transform
saved one output-sized array at a measured 4.3% increase in median time in this
run; the other comparisons were within 3.5%. Allocation regressions are enforced
by tests, while timings remain benchmark comparisons.
