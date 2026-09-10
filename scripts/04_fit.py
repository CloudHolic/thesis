"""Fit the item parameters on real responses, ours or the paper's.

uv run scripts/04_fit.py --loss map
uv run scripts/04_fit.py --reference
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from thesis import artifact, config, domain, report
from thesis.data import dataset
from thesis.model import fit as model_fit
from thesis.model.family.protocol import Family
from thesis.model.family.registry import FAMILIES
from thesis.model.loss.registry import LOSSES
from thesis.utils import precision

SCRIPT = Path(__file__).name
FAMILY_NAMES = tuple(sorted(FAMILIES))
LOSS_NAMES = tuple(sorted(LOSSES))


@dataclass(frozen=True, slots=True)
class Inputs:
	"""One dataset, resolved to the response variable and pool this run asked for."""

	data: dataset.Dataset
	response: np.ndarray
	person: np.ndarray
	train: np.ndarray
	response_name: str
	pool: str
	source: Path


def load_inputs(cfg: config.Config, args: argparse.Namespace) -> Inputs:
	"""Read the pilot dataset and pick out the response this run fits."""
	response_name = args.response or cfg.data.response
	pool = args.pool or cfg.data.pool
	source = cfg.directory("dataset") / f"pilot_{pool}.npz"
	data = dataset.load(source)
	if response_name not in data.responses:
		raise SystemExit(f"{source.name} has no response {response_name!r}")

	return Inputs(
		data=data,
		response=data.responses[response_name],
		person=data.person_index(),
		train=~data.held_out,
		response_name=response_name,
		pool=pool,
		source=source,
	)


def latent_parameters(family: Family, data: dataset.Dataset) -> int:
	"""The item coordinates the run carries."""
	return family.Z_DIM * data.n_items


def environment(cfg: config.Config, args: argparse.Namespace, inputs: Inputs) -> dict[str, Any]:
	"""What the provenance record says about where this run happened."""
	return {
		"family": args.family,
		"response": inputs.response_name,
		"pool": inputs.pool,
		"precision": cfg.model.precision,
		"devices": [str(d) for d in jax.devices()],
		"source": inputs.source.name,
	}


def sizes(inputs: Inputs, latent: int) -> dict[str, int]:
	"""What the provenance record says about the data this run saw."""
	data = inputs.data

	return {
		"n_items": data.n_items,
		"n_persons": data.n_persons,
		"n_obs": data.n_obs,
		"held_out": int(data.held_out.sum()),
		"latent_parameters": latent,
	}


def settings(cfg: config.Config) -> model_fit.Settings:
	"""Everything the optimizer loop reads out of the configuration file."""
	return model_fit.Settings(
		quadrature_nodes=cfg.model.quadrature_nodes,
		optimizer=cfg.train.optimizer,
		learning_rate=cfg.train.learning_rate,
		steps=cfg.train.steps,
		tolerance=cfg.train.tolerance,
		patience=cfg.train.patience,
		seed=cfg.train.seed,
	)


def run_reference(
	cfg: config.Config, args: argparse.Namespace, family: Family, inputs: Inputs
) -> None:
	"""The paper's own model."""
	# Importing numpyro is expensive, and a run without --reference never needs it.
	from thesis.references.zoi import mcmc

	data = inputs.data
	train = inputs.train

	started = time.monotonic()
	post = mcmc.run(
		data.item_index[train],
		inputs.person[train],
		inputs.response[train],
		data.n_items,
		data.n_persons,
		seed=cfg.train.seed,
		chains=cfg.reference.chains,
		samples=cfg.reference.samples,
		warmup=cfg.reference.warmup,
	)
	minutes = (time.monotonic() - started) / 60.0

	worst = {site: float(np.nanmax(v)) for site, v in post.r_hat.items()}
	body = report.table(["site", "max r_hat"], [[f"`{s}`", f"{v:.4f}"] for s, v in worst.items()])
	print(
		f"nuts      {minutes:.1f} min, divergences {post.divergences:,}, "
		f"max r_hat {max(worst.values()):.4f}\n\n{body}"
	)

	sites = list(family.SITES)
	latent = latent_parameters(family, data)
	stem = f"reference_{args.family}_{inputs.response_name}_{inputs.pool}"
	out = cfg.directory("fits") / f"{stem}.npz"
	artifact.write(
		out,
		artifact.Artifact(
			arrays={
				"sites": np.array(sites),
				"mean": np.stack([post.mean[s] for s in sites]),
				"median": np.stack([post.median[s] for s in sites]),
				"lower": np.stack([post.lower[s] for s in sites]),
				"upper": np.stack([post.upper[s] for s in sites]),
				"r_hat": np.stack([post.r_hat[s] for s in sites]),
			},
			title=f"Reference NUTS on `{inputs.response_name}`",
			preamble=(
				f"{data.n_items:,} items, {data.n_persons:,} persons, "
				f"{latent:,} latent parameters. "
				f"{minutes:.1f} minutes, {post.divergences:,} divergences."
			),
			body=body,
			extra={
				"effective": {**environment(cfg, args, inputs), "inference": "reference"},
				"dataset": sizes(inputs, latent),
				"nuts": {
					"minutes": round(minutes, 1),
					"divergences": post.divergences,
					"max_r_hat": max(worst.values()),
					"r_hat_by_site": worst,
				},
			},
		),
		script=SCRIPT,
		config_raw=cfg.raw,
	)
	print(f"\nwrote {out}")


def run_fit(cfg: config.Config, args: argparse.Namespace, family: Family, inputs: Inputs) -> None:
	"""Our own inference."""
	data = inputs.data
	result = model_fit.run(
		LOSSES[args.loss](family),
		family=family,
		item_index=data.item_index,
		person_index=inputs.person,
		response=inputs.response,
		n_items=data.n_items,
		n_persons=data.n_persons,
		held_out=data.held_out,
		settings=settings(cfg),
	)

	usable = int(result.theta_usable.sum())
	print(
		f"fit       {result.steps} steps, first {result.seconds[0]:.1f}s incl. compile, "
		f"then {1000 * result.step_seconds:.1f} ms/step, converged={result.converged}"
	)
	print(f"holdout   {result.held_out_log_likelihood:.4f} log-likelihood per cell")
	print(f"centres   {usable:,}/{data.n_persons:,} adaptive")

	shares = {name: float(flag.mean()) for name, flag in result.extrapolated.items()}
	body = report.distribution(result.quantities)
	flags = report.shares(shares, ["quantity", "extrapolated"])
	print(f"\n{body}\n\n{flags}")

	flag_names = sorted(result.extrapolated)
	reported_names = sorted(result.reported)
	latent = latent_parameters(family, data)
	stem = f"{args.family}_{args.loss}_{inputs.response_name}_{inputs.pool}"
	out = cfg.directory("fits") / f"{stem}.npz"
	artifact.write(
		out,
		artifact.Artifact(
			arrays={
				"reported_names": np.array(reported_names),
				"reported": np.stack([result.reported[name] for name in reported_names]),
				"losses": result.losses,
				"seconds": result.seconds,
				"theta_mode": result.theta_mode,
				"theta_usable": result.theta_usable,
				"item_theta_low": result.item_theta_low,
				"item_theta_high": result.item_theta_high,
				"flag_names": np.array(flag_names),
				"extrapolated": np.stack([result.extrapolated[name] for name in flag_names]),
			},
			title=f"`{args.loss}` on `{inputs.response_name}`",
			preamble=(
				f"{data.n_items:,} items, {data.n_persons:,} persons, "
				f"{int(inputs.train.sum()):,} training cells. "
				f"Held-out log-likelihood {result.held_out_log_likelihood:.4f} per cell."
			),
			body=f"{body}\n\n{flags}",
			extra={
				"effective": {**environment(cfg, args, inputs), "inference": args.loss},
				"dataset": sizes(inputs, latent),
				"fit": {
					"steps": result.steps,
					"converged": result.converged,
					"best_loss": result.best_loss,
					"first_step_seconds": round(float(result.seconds[0]), 2),
					"ms_per_step": round(1000 * result.step_seconds, 2),
					"adaptive_centres": usable,
				},
				"holdout_log_likelihood_per_cell": result.held_out_log_likelihood,
				"extrapolated": shares,
			},
		),
		script=SCRIPT,
		config_raw=cfg.raw,
	)
	print(f"\nwrote {out}")


def parse_args() -> argparse.Namespace:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--config", type=Path, default=None, help="path to config.toml")
	ap.add_argument("--response", default=None, choices=sorted(domain.VIEWS))
	ap.add_argument("--pool", default=None, choices=list(domain.POOLS))
	ap.add_argument("--family", default="zoi_beta", choices=FAMILY_NAMES)
	ap.add_argument("--loss", default="map", choices=LOSS_NAMES)
	ap.add_argument("--reference", action="store_true", help="fit the paper's model instead")

	return ap.parse_args()


def main() -> None:
	args = parse_args()
	cfg = config.load(args.config)
	precision.enable(cfg.model.precision)

	family = FAMILIES[args.family]
	inputs = load_inputs(cfg, args)
	data = inputs.data
	print(
		f"data      {data.n_items:,} items, {data.n_persons:,} persons, {data.n_obs:,} "
		f"responses, {int(data.held_out.sum()):,} held out, "
		f"{latent_parameters(family, data):,} latent"
	)

	if args.reference:
		run_reference(cfg, args, family, inputs)
	else:
		run_fit(cfg, args, family, inputs)


if __name__ == "__main__":
	main()
