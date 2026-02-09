# reject-inference-lendingclub
Reject Inference, to be specific, investigate the possibility of using statistical methods (extrapolation and NB) and supervised learning algorithms (LightGBM, xgboost, RF, LR, catboost) to build a benchmark model through the Ensemble Learning (Weighted Voting) to predict the default status of rejected samples and therefore obtain full sample data set. 
One challenge of this research is proposing a new training sample selection process, which requires an effective mechanism for rejecting the sample inclusion ratio and multiple rounds of iterative verification based on AUC. 

## QUBO-based reject selection

The `qubo_selection.py` module provides helpers for constructing the QUBO used to
select rejected applications for labeling each round. The workflow matches the
"uncertainty + diversity + budget" formulation:

1. Train an ensemble on current labeled data.
2. Compute per-reject utilities via ensemble disagreement.
3. Build a similarity matrix from standardized features.
4. Assemble QUBO coefficients and solve for the labeling set.

Example usage:

```python
import numpy as np
from qubo_selection import build_qubo, compute_disagreement, compute_similarity

# probabilities: shape (n_samples, n_models)
utilities = compute_disagreement(probabilities, method="variance")

# features: shape (n_samples, n_features)
similarities = compute_similarity(features, metric="cosine")

terms = build_qubo(
    utilities=utilities,
    similarities=similarities,
    budget=500,
    diversity_lambda=0.5,
    penalty=10.0,
)
```
