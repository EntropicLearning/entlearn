# Test organisation

Group collected tests as methods of purpose-named `Test...` classes. This applies
throughout `tests/`, including subprocess tests. Keep fixtures and
reusable helpers at module level; put shared test data in the relevant `_fixtures.py`.

When regrouping tests, preserve their parameterisation, assertions and subprocess
programs. Compare collection counts before and after, then run the affected tests.

A refusal test's `match=` names the offending value and the shortest phrase of the reason.
Ruff's `PT011` fails a broad `pytest.raises` without one, and `tests/conftest.py` fails a
pattern that spells out a whole message.

Pytest turns every warning into an error. A test whose subject is a warning asserts it with
`pytest.warns(Category, match=...)`, under the same `match=` rule. A warning that a test's
setup causes but does not test is filtered on that test: `FIXED_BUDGET` from `conftest.py`
for a fit that runs a fixed number of iterations, `RANK_DEFICIENT` for a small manifold
fixture, and otherwise `@pytest.mark.filterwarnings("ignore:<message start>:<Category>")`.
`pyproject.toml` filters only third-party warnings that no call site can avoid.

## Running the checks

While working, run the test file you changed, then its directory. Before opening a pull
request, run `scripts/gate` once: it runs Ruff, `ty`, the checks in `scripts/` and this suite.
A change to numerical code also runs its directory with `EON_TEST_DTYPE=float32`.
[scripts/README.md](../scripts/README.md) lists the `scripts/` checks and what each guards.

## Lanes and depth

Four environment variables, read once by `tests/conftest.py`, select what a run covers:

- `EON_TEST_DEVICE`: `cpu` (default) or `cuda`.
- `EON_TEST_DTYPE`: `float64` or `float32`; the default is the device's own dtype, `float64`
  on cpu and `float32` on cuda.
- `EON_TEST_DEPTH`: `minimal`, `standard` (default) or `exhaustive`. It scales the seed
  replicates: one seed per seed axis at `minimal`, the counts CI runs at `standard`, and
  more seeds at `exhaustive`.
- `EON_TEST_DETERMINISTIC`: `1` (or `true`, `yes`, `on`) runs the session with
  `torch.use_deterministic_algorithms(True)`. Without it every lane runs in PyTorch's default
  mode, where a fit repeats bitwise from run to run on CUDA too, so the identity tests check
  what users run. The flag adds two checks the default mode lacks: every new `torch.empty`
  tensor is filled with NaN, so a read of uninitialised scratch shows, and an operation with
  no deterministic implementation raises. Run the cuda lane with it once before a release.

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

References follow the package's three numerical roles: a temperature
of zero is hard and any positive one soft, in every dtype; masses and probability logs use
`finfo(dtype).tiny`; tolerances use `eps`. Entropy references use exact positive logarithms
with `0 log 0 = 0`. A test of a regime or emptiness boundary places values on both sides of
it, at zero or `tiny`, not at `eps`. For contributions below `eps`, use relative error or an
absolute bound derived from the contribution's own scale, so an ordinary absolute tolerance
cannot hide a regression to an `eps` floor.
