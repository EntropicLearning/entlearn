# Glossary

These terms describe EON's representations, learning and prediction. Package objects and interfaces are introduced in [Package and Recipes](../guide/package.md).

| Term | Meaning |
| --- | --- |
| **Affiliation** | An observation's membership distribution over a block's clusters, concentrated in one cluster for hard assignments or spread across clusters for soft assignments. See [Affiliations](representation.md#observations-and-affiliations) |
| **Centroid** | A cluster's representative point or categorical distribution in input space, or its representative target vector for regression. See [Input geometry](representation.md#standard-input-geometry) and [output centroids](representation.md#regression-head) |
| **Discretisation error** | The error of representing an observation by an input cluster's geometry. See [Standard error](representation.md#the-discretisation-error) and [manifold error](representation.md#manifold-geometry) |
| **Transition matrix** | A matrix relating source clusters to target clusters or classes. See [Transitions](representation.md#transitions-between-clusterings) |
| **Coupling** | A transition's normalisation constraint, with columns summing to one under M and rows summing to one under S. See [Coupling conventions](representation.md#transitions-between-clusterings) |
| **Weights** | Probability vectors controlling contributions of features, training observations or regression output dimensions to the loss. See [Feature, instance and output weights](representation.md#feature-instance-and-output-weights) |
| **Entropy reward** | A term subtracted from the loss to favour spread-out affiliations or learned weights. See [Entropy rewards](loss.md#entropy-rewards) |
| **Temperature** | A coefficient balancing entropy against cost. See [Affiliation regimes](learning.md#affiliation-temperatures) and [weight regimes](learning.md#temperatures-and-effective-dimension) |
| **Effective dimension** | An entropy-based measure of how broadly a distribution spreads its weight. See [Effective dimension](learning.md#temperatures-and-effective-dimension) |
| **Pruning** | Removal of clusters with numerically zero unweighted affiliation mass and their associated parameters. See [Pruning](learning.md#pruning) |
| **Read-out** | The rule converting final affiliations into a prediction. See [Read-outs](prediction.md#reading-out-a-prediction) |
| **Iterative prediction** | Repeated updates of query affiliations and prediction coordinates, including recovered instance weights when available, with learned parameters fixed. See [Iterative prediction](prediction.md#iterative-prediction) |
