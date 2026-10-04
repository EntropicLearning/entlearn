# Test organisation

Group collected tests as methods of purpose-named `Test...` classes. This applies
throughout `tests/`, including subprocess tests. Keep fixtures and
reusable helpers at module level; put shared test data in the relevant `_fixtures.py`.

When regrouping tests, preserve their parameterisation, assertions and subprocess
programs. Compare collection counts before and after, then run the affected tests.

## On-demand checks in `scripts/`

`scripts/` holds checks outside the ordinary `tests/` collection;
[scripts/README.md](../scripts/README.md) lists them and when to run each.

## Lanes and depth

Four environment variables, read once by `tests/conftest.py`, select what a run covers:

- `EON_TEST_DEVICE`: `cpu` (default) or `cuda`.
- `EON_TEST_DTYPE`: `float64` or `float32`; the default is the device's own dtype, `float64`
  on cpu and `float32` on cuda.
- `EON_TEST_DEPTH`: `minimal`, `standard` (default) or `exhaustive`. It scales the seed
  replicates: one seed per seed axis at `minimal`, the counts CI runs at `standard`, and
  more seeds at `exhaustive`.
- `EON_TEST_DETERMINISTIC`: `1` (or `true`, `yes`, `on`) runs the session with
  `torch.use_deterministic_algorithms(True)`. The cuda lane always does, because CUDA's
  atomic `index_add_` is not bitwise reproducible from run to run.

The header line `EON test backend: ...` reports the active lane and depth, and whether
deterministic algorithms are enforced. A device, dtype or depth the run cannot honour aborts
the session with a usage error rather than falling back.

The seeds are fixed. Each depth's seeds are a prefix of the next depth's, so a test ID
names the same case at every depth and an `exhaustive` failure reproduces by ID at that
depth. Scale a new seed axis with `conftest.by_depth(minimal, standard, exhaustive)`. A
test that needs a single seed reads `SEEDS[0]` and takes a second independent stream from
`SEEDS[0] + 1`, so it runs unchanged at `minimal`.

The mutation sweep runs at `minimal`; the survivor confirmation pass runs at `standard`,
so the score reflects the tests CI runs.

## Feature-data regime

Ordinary continuous feature data used for fitting, initialisation and prediction
is in or near `[0, 1]`. Generate or scale synthetic fixtures accordingly, applying
the same feature transform to training rows, query rows and supplied geometry.
Do not change categorical codes, labels, regression targets or abstract
mathematical operands merely to match this feature convention.

Native scikit-learn checks generate their own, potentially unscaled data and
exercise the estimators' default float64 configuration. Precision-specific
adapter tests use scaled features. The tests do not require protection against
numerical stress from unscaled input.

## Numerical comparisons

Approximate comparisons use tolerances appropriate to the actual computation
dtype and the operation being checked. Derive them from `torch.finfo(dtype).eps`
or `np.finfo(array.dtype).eps`, or use an existing dtype-aware comparison helper.
Use the computed result's dtype, not an original DataFrame's storage dtype or a
global test dtype when the test explicitly overrides it. Explain the error budget
when the operation needs more than ordinary rounding allowance.

Specify both relative and absolute tolerances for NumPy comparisons and
`torch.allclose`: a dtype-aware absolute tolerance does not cancel their
dtype-insensitive default relative tolerance. For `pytest.approx`, supply
dtype-appropriate `rel` and/or `abs` explicitly; an `abs`-only call intentionally
uses only the absolute bound. `torch.testing.assert_close` has dtype-aware
defaults; retain tighter operation-specific bounds where the contract requires
them.

Exact equality remains appropriate for labels, row routing, unchanged storage,
pure delegation and exact dtype conversions. Do not weaken those checks merely
to introduce a tolerance.

Fixed-dtype reference comparisons may use fixed, justified tolerances when the
dtype is explicitly pinned. Statistical estimation or deliberately approximate
algorithms may need an error bound beyond floating-point round-off; distinguish
and explain that bound. Do not loosen existing monotonicity or other scientific
tolerances to make a failing test pass.
