"""Standalone uncertainty analysis: copy only this file beside your script.

Requires Python 3.9+, NumPy, and SciPy. PyTorch is needed only for tensor
inputs; Matplotlib is needed only for plotting. Importing performs no file
I/O, model downloads, or plotting. Scores are evidence of correctness,
not calibrated correctness probabilities.

Example::

    import uncertainty_analysis as ua

    margins = ua.delta_top1_logsumexp_rest([[4.0, 1.0, 0.0]])
    model = ua.KDEModel.fit(
        [-2.0, -1.0, 1.0, 2.0], [False, False, True, True],
    )
    scores = model.log_likelihood_ratio(margins)

This file is self-contained and does not import the uncertainty_analysis/
package. It also supports `python uncertainty_analysis.py --help`.
"""

import argparse
import json
import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp
from scipy.stats import gaussian_kde

__version__ = "0.1.0"
__all__ = [
    "KDEModel", "MCQScores", "aggregate_token_ratios",
    "delta_top1_logsumexp_rest", "kde_pdf", "load_jsonl",
    "plot_densities", "score_analysis_records", "score_mcq_records",
]



def delta_top1_logsumexp_rest(logits, T: float = 1.0):
    """Return ``top1(logits/T) - logsumexp(rest(logits/T))``.

    Accept shape ``(..., classes)`` with at least two classes. Return a NumPy
    array (or scalar for one row). PyTorch inputs return tensors on the same
    device and retain autograd; PyTorch is imported only for tensor inputs.
    Half-precision and integer tensors are upcast to float32.

    Raw logits and normalized log probabilities give the same margin. Negative
    infinity may mask classes; every row needs a finite class. A row with only
    one unmasked class has an infinite positive margin.
    """
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be finite and positive")
    if any(cls.__module__.startswith("torch") for cls in type(logits).__mro__):
        return _torch_margin(logits, T=T)
    values = np.asarray(logits, dtype=float)
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("logits must have shape (..., classes) with at least two classes")
    if np.any(np.isnan(values) | np.isposinf(values)):
        raise ValueError("logits cannot contain NaN or positive infinity")
    if not np.all(np.any(np.isfinite(values), axis=-1)):
        raise ValueError("each row must contain a finite logit")
    # Center before scaling to avoid overflow from a large common offset.
    centered = (values - np.max(values, axis=-1, keepdims=True)) / float(T)
    indices = np.argmax(centered, axis=-1, keepdims=True)
    rest = centered.copy()
    np.put_along_axis(rest, indices, -np.inf, axis=-1)
    return -logsumexp(rest, axis=-1)


def _torch_margin(logits, T: float = 1.0):
    """Compute the margin on the last dimension, preserving device and gradients."""
    import torch

    if not math.isfinite(T) or T <= 0:
        raise ValueError("T must be finite and positive")
    if logits.ndim < 1 or logits.shape[-1] < 2:
        raise ValueError("logits must have shape (..., classes) with at least two classes")
    if logits.is_complex():
        raise ValueError("logits must be real")
    z = logits
    if not z.is_floating_point() or z.dtype in (torch.float16, torch.bfloat16):
        z = z.float()
    if (torch.isnan(z) | torch.isposinf(z)).any().item():
        raise ValueError("logits cannot contain NaN or positive infinity")
    if not torch.isfinite(z).any(dim=-1).all().item():
        raise ValueError("each row must contain a finite logit")
    maximum, indices = z.max(dim=-1, keepdim=True)
    centered = (z - maximum) / float(T)
    rest = centered.scatter(-1, indices, -float("inf"))
    return -torch.logsumexp(rest, dim=-1)


@dataclass(frozen=True)
class MCQScores:
    """Parallel one-dimensional arrays of margins, correctness, predictions."""

    margins: np.ndarray
    correct: np.ndarray
    predictions: np.ndarray


def load_jsonl(path):
    """Load JSON objects from UTF-8 JSONL, skipping blank lines."""
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}: line {number} must contain a JSON object")
            records.append(record)
    return records


def score_mcq_records(
    records: Iterable[Mapping], options: Sequence[str] = ("A", "B", "C", "D"),
    *, T: float = 1.0,
) -> MCQScores:
    """Score records with option_logprobs and gt fields.

    gt is a zero-based index into options or an option label. Prediction is
    recomputed from scores (ties follow options order), ignoring cached pred
    and ok fields. Every option needs a score. Margins are conditional on these
    options, not the model's full vocabulary.
    """
    options = tuple(options)
    if len(options) < 2 or len(set(options)) != len(options):
        raise ValueError("options must contain at least two distinct labels")
    rows, labels = [], []
    for index, record in enumerate(records):
        try:
            scores = record["option_logprobs"]
            if not isinstance(scores, Mapping) or set(scores) != set(options):
                raise ValueError("option_logprobs keys must match options exactly")
            row = [float(scores[option]) for option in options]
            gt = record["gt"]
            if isinstance(gt, Integral) and not isinstance(gt, (bool, np.bool_)):
                if not 0 <= gt < len(options):
                    raise ValueError("gt index is outside options")
                gt = options[gt]
            if gt not in options:
                raise ValueError("gt must be an option label or zero-based index")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid MCQ record {index}: {exc}") from exc
        rows.append(row)
        labels.append(gt)
    matrix = np.asarray(rows, dtype=float).reshape(-1, len(options))
    margins = delta_top1_logsumexp_rest(matrix, T=T)
    predictions = np.asarray(options)[np.argmax(matrix, axis=-1)]
    return MCQScores(margins, predictions == np.asarray(labels), predictions)


def _samples(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"{name} needs at least two one-dimensional samples")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} samples must be finite")
    if np.all(values == values[0]):
        raise ValueError(f"{name} samples must have nonzero variance")
    return values


def _fit(values, bandwidth, name):
    values = _samples(values, name)
    if bandwidth is not None and not isinstance(bandwidth, str):
        if not np.isscalar(bandwidth) or not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth must be 'scott', 'silverman', or a positive scalar")
    if isinstance(bandwidth, str) and bandwidth not in ("scott", "silverman"):
        raise ValueError("bandwidth must be 'scott', 'silverman', or a positive scalar")
    try:
        return gaussian_kde(values, bw_method=bandwidth)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"cannot fit {name} KDE: {exc}") from exc


def _points(values):
    points = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(points)):
        raise ValueError("evaluation scores must be finite")
    return points


def kde_pdf(vals, xs, bw="scott"):
    """Return (density, fitted_kde) on xs, always using the same return type.

    Nonfinite training values are discarded, matching the original helper.
    Fewer than two finite samples or constant samples raise ValueError, rather
    than returning an inconsistent type or adding random jitter.
    """
    vals = np.asarray(vals, dtype=float)
    if vals.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    kde = _fit(vals[np.isfinite(vals)], bw, "density")
    points = _points(xs)
    if not points.size:
        return np.empty(points.shape), kde
    return kde(points.reshape(-1)).reshape(points.shape), kde


class KDEModel:
    """Two one-dimensional Gaussian KDEs conditioned on correctness.

    Fit only on reference/training scores. A density ratio above one favors
    the correct class under equal priors; it is not a correctness probability.
    """

    def __init__(self, kde_correct, kde_incorrect):
        for name, kde in (("correct", kde_correct), ("incorrect", kde_incorrect)):
            if not isinstance(kde, gaussian_kde) or kde.d != 1:
                raise ValueError(f"{name} must be a one-dimensional scipy gaussian_kde")
        self.kde_correct = kde_correct
        self.kde_incorrect = kde_incorrect

    @classmethod
    def fit(cls, scores, correct, *, bandwidth="scott"):
        """Fit separate KDEs, requiring two distinct finite scores per class."""
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(correct)
        if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
            raise ValueError("scores and correct must be aligned one-dimensional arrays")
        if not np.all(np.isin(labels, [False, True])):
            raise ValueError("correct labels must be booleans or 0/1")
        labels = labels.astype(bool)
        return cls(_fit(scores[labels], bandwidth, "correct"),
                   _fit(scores[~labels], bandwidth, "incorrect"))

    def densities(self, scores):
        """Return (correct_pdf, incorrect_pdf) with the input's shape."""
        points = _points(scores)
        if not points.size:
            return np.empty(points.shape), np.empty(points.shape)
        flat = points.reshape(-1)
        return (self.kde_correct(flat).reshape(points.shape),
                self.kde_incorrect(flat).reshape(points.shape))

    def log_likelihood_ratio(self, scores):
        """Return log p(score|correct) - log p(score|incorrect).

        Log-space evaluation avoids division of underflowed PDFs. Extremely
        remote scores whose log densities exceed float range raise ValueError.
        """
        points = _points(scores)
        if not points.size:
            return np.empty(points.shape)
        flat = points.reshape(-1)
        with np.errstate(over="ignore", invalid="ignore"):
            ratio = self.kde_correct.logpdf(flat) - self.kde_incorrect.logpdf(flat)
        if not np.all(np.isfinite(ratio)):
            raise ValueError("scores are too extreme for finite log-density ratios")
        return ratio.reshape(points.shape)

    def likelihood_ratio(self, scores):
        """Return p(score|correct)/p(score|incorrect); may be 0 or inf.

        Prefer log_likelihood_ratio for ranking very extreme scores.
        """
        with np.errstate(over="ignore", under="ignore"):
            return np.exp(self.log_likelihood_ratio(scores))

    def to_legacy_dict(self):
        """Return the kde_models.pkl notebook dictionary layout."""
        return {"kde_correct": self.kde_correct, "kde_incorrect": self.kde_incorrect}

    @classmethod
    def from_legacy_dict(cls, models):
        """Wrap KDEs from an already loaded, trusted notebook model dictionary."""
        return cls(models["kde_correct"], models["kde_incorrect"])

    def save(self, path):
        """Save datasets, weights, and fitted bandwidth factors as UTF-8 JSON."""
        def encode(kde):
            return {"samples": kde.dataset[0].tolist(),
                    "weights": kde.weights.tolist(), "factor": float(kde.factor)}
        payload = {"format": "uncertainty-analysis-kde", "version": 1,
                   "correct": encode(self.kde_correct),
                   "incorrect": encode(self.kde_incorrect)}
        Path(path).write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        """Load a JSON model created by save; no pickle deserialization."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format") != "uncertainty-analysis-kde" or payload.get("version") != 1:
            raise ValueError("unsupported uncertainty-analysis model format")

        def decode(name):
            try:
                entry = payload[name]
                samples = _samples(entry["samples"], name)
                weights = np.asarray(entry["weights"], dtype=float)
                factor = float(entry["factor"])
                if weights.shape != samples.shape or not np.all(np.isfinite(weights)):
                    raise ValueError("invalid weights")
                if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
                    raise ValueError("weights must be nonnegative and sum to one")
                if not np.isfinite(factor) or factor <= 0:
                    raise ValueError("invalid bandwidth factor")
                return gaussian_kde(samples, weights=weights, bw_method=factor)
            except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
                raise ValueError(f"invalid {name} KDE model: {exc}") from exc
        return cls(decode("correct"), decode("incorrect"))


def aggregate_token_ratios(tokens, *, log=False):
    """Return the minimum stored kde_correct / kde_incorrect token ratio.

    Higher scores indicate stronger evidence of correctness. Empty sequences,
    invalid densities, and 0/0 fail explicitly. A positive numerator over zero
    gives infinity. For newly fitted models use score_analysis_records with
    a model to avoid density underflow. log=True computes differences of log
    densities directly, avoiding overflow of otherwise finite ratios.
    """
    ratios = []
    for index, token in enumerate(tokens):
        correct = float(token["kde_correct"])
        incorrect = float(token["kde_incorrect"])
        if not np.isfinite(correct) or not np.isfinite(incorrect) or min(correct, incorrect) < 0:
            raise ValueError(f"token {index}: densities must be finite and nonnegative")
        if correct == incorrect == 0:
            raise ValueError(f"token {index}: both densities are zero; use a fitted model in log space")
        if log:
            with np.errstate(divide="ignore"):
                ratios.append(float(np.log(correct) - np.log(incorrect)))
        else:
            ratios.append(correct / incorrect if incorrect else float("inf"))
    if not ratios:
        raise ValueError("cannot score an empty token sequence")
    return float(min(ratios))


def score_analysis_records(records, model=None, *, log=False):
    """Return (scores, correct) arrays for notebook analysis result records.

    Each record has pred, gt, and a nonempty prob_each_token sequence. Without
    model, use stored KDE densities. With model, use each token's token_score
    and aggregate log ratios before exponentiating. log=True returns minimum
    log ratios for stable ranking. Labels use the original exact pred == gt.
    """
    scores, labels = [], []
    for index, record in enumerate(records):
        try:
            tokens = record["prob_each_token"]
            if not tokens:
                raise ValueError("cannot score an empty token sequence")
            if model is None:
                value = aggregate_token_ratios(tokens, log=log)
            else:
                token_scores = [token["token_score"] for token in tokens]
                value = float(np.min(model.log_likelihood_ratio(token_scores)))
                if not log:
                    with np.errstate(over="ignore", under="ignore"):
                        value = float(np.exp(value))
            labels.append(record["pred"] == record["gt"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid analysis record {index}: {exc}") from exc
        scores.append(value)
    return np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)


def plot_densities(model, xs=None, *, ax=None):
    """Plot the two fitted PDFs and return an Axes without calling show()."""
    import matplotlib.pyplot as plt

    if xs is None:
        samples = np.concatenate((model.kde_correct.dataset[0], model.kde_incorrect.dataset[0]))
        low, high = samples.min(), samples.max()
        padding = 0.02 * (high - low) if high > low else 1.0
        xs = np.linspace(low - padding, high + padding, 512)
    xs = np.asarray(xs, dtype=float)
    if xs.ndim != 1 or xs.size == 0:
        raise ValueError("xs must be a nonempty one-dimensional grid")
    correct, incorrect = model.densities(xs)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    ax.plot(xs, correct, label="Correct", linewidth=2)
    ax.plot(xs, incorrect, label="Incorrect", linewidth=2)
    ax.set_xlabel("Top-one vs. remaining-class log odds")
    ax.set_ylabel("PDF (KDE)")
    ax.legend()
    return ax


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit and apply uncertainty density models")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit correct/incorrect KDEs from MCQ JSONL")
    fit.add_argument("input", type=Path)
    fit.add_argument("output", type=Path, help="output JSON model")
    fit.add_argument("--temperature", type=float, default=1.0)
    fit.add_argument("--options", nargs="+", default=["A", "B", "C", "D"])
    fit.add_argument("--bandwidth", choices=["scott", "silverman"], default="scott")
    fit.add_argument("--plot", type=Path, help="optional PDF/PNG plot (requires Matplotlib)")
    score = commands.add_parser("score", help="score analysis records from JSONL")
    score.add_argument("input", type=Path)
    score.add_argument("output", type=Path, help="output JSON score arrays")
    score.add_argument("--model", type=Path, help="use token_score fields with this JSON model")
    score.add_argument("--log", action="store_true", help="return log likelihood ratios")
    args = parser.parse_args(argv)
    inputs = [args.input]
    if args.command == "score" and args.model:
        inputs.append(args.model)
    outputs = [args.output]
    if args.command == "fit" and args.plot:
        outputs.append(args.plot)
    if len({p.resolve() for p in inputs + outputs}) != len(inputs + outputs):
        parser.error("input, model, output, and plot paths must be distinct")
    try:
        records = load_jsonl(args.input)
        if args.command == "fit":
            result = score_mcq_records(records, args.options, T=args.temperature)
            model = KDEModel.fit(result.margins, result.correct, bandwidth=args.bandwidth)
            if args.plot:
                ax = plot_densities(model)
                ax.figure.tight_layout()
                ax.figure.savefig(args.plot)
            model.save(args.output)
        else:
            model = KDEModel.load(args.model) if args.model else None
            values, correct = score_analysis_records(records, model, log=args.log)
            if not np.all(np.isfinite(values)):
                raise ValueError("output contains infinite ratios; score with --model and --log")
            payload = {"scores": values.tolist(), "correct": correct.tolist(),
                       "score_type": "log_likelihood_ratio" if args.log else "likelihood_ratio"}
            args.output.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
