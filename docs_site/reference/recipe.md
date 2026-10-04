# Recipe

A `Recipe` describes the blocks, connections and settings of a model.

Two public aliases name a block's role in type annotations: `entlearn.InputBlock` is
`Input | ManifoldInput`, and `entlearn.Head` is `ClassificationHead | RegressionHead`.
Both aliases are unions of the block description types listed below.

::: entlearn.Recipe

::: entlearn.Input

::: entlearn.ManifoldInput

::: entlearn.Hidden

::: entlearn.ClassificationHead

::: entlearn.RegressionHead

::: entlearn.Connection

::: entlearn.Coupling
