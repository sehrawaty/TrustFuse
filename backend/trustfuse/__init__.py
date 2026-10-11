"""TrustFuse: security evaluation of intentional modality suppression in wearable health AI.

The package mirrors the software design of the project report:

* ``data``          -> DatasetLoader, Preprocessor
* ``attacks``       -> AttackSimulator (Selective Modality Suppression Attack, SMSA)
* ``models``        -> FusionModel, unimodal baselines, trainer
* ``reliability``   -> ReliabilityEngine (signal quality Q, disagreement D, uncertainty U -> R)
* ``defense``       -> reliability-aware fusion
* ``evaluation``    -> Evaluator, statistics, hypotheses H1-H4, paper figures
* ``experiments``   -> ExperimentConfig, runner, registry
* ``service``/``api`` -> inference service and FastAPI application
"""

__version__ = "1.0.0"
