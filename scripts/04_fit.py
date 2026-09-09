"""Fit the item parameters on real responses, ours or the paper's.

uv run scripts/04_fit.py --loss zoi_map
uv run scripts/04_fit.py --reference
"""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path
from typing import cast

import jax
import numpy as np

from thesis import config, domain, report, runmeta
from thesis.data import dataset
from thesis.model import fit as model_fit
from thesis.model import precision

LOSSES = ("zoi_map", "zoi_elbo")


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--config", type=Path, default=None, help="path to config.toml")
	ap.add_argument("--response", default=None, choices=sorted(domain.VIEWS))
	ap.add_argument("--pool", default=None, choices=list(domain.POOLS))
	ap.add_argument("--loss", default="zoi_map", choices=LOSSES)
	ap.add_argument("--reference", action="store_true", help="fit the paper's model instead")
	args = ap.parse_args()

	cfg = config.load(args.config)
	precision.enable(cfg.model.precision)

	response_name = args.response or cfg.data.response
	pool = args.pool or cfg.data.pool
	source = cfg.directory("dataset") / f"pilot_{pool}.npz"
	data = dataset.load(source)
	if response_name not in data.responses:
		raise SystemExit(f"{source.name} has no response {response_name!r}")

	response = data.responses[response_name]
	person = data.person_index()
	train = ~data.held_out
	latent = data.n_persons + 5 * data.n_items
	print(
		f"data      {data.n_items:,} items, {data.n_persons:,} persons, {data.n_obs:,} "
		f"responses, {int(data.held_out.sum()):,} held out, {latent:,} latent"
	)

	fits = cfg.directory("fits")
	environment = {
		"response": response_name,
		"pool": pool,
		"precision": cfg.model.precision,
		"devices": [str(d) for d in jax.devices()],
		"source": source.name,
	}
	sizes = {
		"n_items": data.n_items,
		"n_persons": data.n_persons,
		"n_obs": data.n_obs,
		"held_out": int(data.held_out.sum()),
		"latent_parameters": latent,
	}

	if args.reference:
		from thesis.references import zoi

		started = time.monotonic()
		post = zoi.run(
			data.item_index[train],
			person[train],
			response[train],
			data.n_items,
			data.n_persons,
			seed=cfg.train.seed,
			chains=cfg.reference.chains,
			samples=cfg.reference.samples,
			warmup=cfg.reference.warmup,
		)
		minutes = (time.monotonic() - started) / 60.0
		worst = {site: float(np.nanmax(v)) for site, v in post.r_hat.items()}
		body = report.table(
			["site", "max r_hat"], [[f"`{s}`", f"{v:.4f}"] for s, v in worst.items()]
		)
		print(
			f"nuts      {minutes:.1f} min, divergences {post.divergences:,}, "
			f"max r_hat {max(worst.values()):.4f}\n\n{body}"
		)

		sites = list(zoi.SITES)
		out = fits / f"reference_{response_name}_{pool}.npz"
		np.savez_compressed(
			out,
			sites=np.array(sites),
			mean=np.stack([post.mean[s] for s in sites]),
			median=np.stack([post.median[s] for s in sites]),
			lower=np.stack([post.lower[s] for s in sites]),
			upper=np.stack([post.upper[s] for s in sites]),
			r_hat=np.stack([post.r_hat[s] for s in sites]),
		)
		out.with_suffix(".md").write_text(
			f"# Reference NUTS on `{response_name}`\n\n{sizes['n_items']:,} items, "
			f"{sizes['n_persons']:,} persons, {latent:,} latent parameters. "
			f"{minutes:.1f} minutes, {post.divergences:,} divergences.\n\n{body}\n",
			encoding="utf-8",
		)
		runmeta.write(
			out,
			runmeta.build(
				script="04_fit.py",
				config_raw=cfg.raw,
				extra={
					"effective": {**environment, "inference": "reference"},
					"dataset": sizes,
					"nuts": {
						"minutes": round(minutes, 1),
						"divergences": post.divergences,
						"max_r_hat": max(worst.values()),
						"r_hat_by_site": worst,
					},
				},
			),
		)
		print(f"\nwrote {out}")
		return

	# A module cannot be checked against a Protocol, so the contract is asserted here
	# and enforced by the loss modules all exposing init, loss and summary.
	objective = cast(
		model_fit.Objective, cast(object, importlib.import_module(f"thesis.model.loss.{args.loss}"))
	)
	result = model_fit.run(
		objective,
		item_index=data.item_index,
		person_index=person,
		response=response,
		n_items=data.n_items,
		n_persons=data.n_persons,
		held_out=data.held_out,
		settings=model_fit.Settings(
			quadrature_nodes=cfg.model.quadrature_nodes,
			optimizer=cfg.train.optimizer,
			learning_rate=cfg.train.learning_rate,
			steps=cfg.train.steps,
			tolerance=cfg.train.tolerance,
			patience=cfg.train.patience,
			seed=cfg.train.seed,
		),
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
	out = fits / f"{args.loss}_{response_name}_{pool}.npz"
	np.savez_compressed(
		out,
		reported_names=np.array(reported_names),
		reported=np.stack([result.reported[name] for name in reported_names]),
		losses=result.losses,
		seconds=result.seconds,
		theta_mode=result.theta_mode,
		theta_usable=result.theta_usable,
		item_theta_low=result.item_theta_low,
		item_theta_high=result.item_theta_high,
		flag_names=np.array(flag_names),
		extrapolated=np.stack([result.extrapolated[name] for name in flag_names]),
	)
	out.with_suffix(".md").write_text(
		f"# `{args.loss}` on `{response_name}`\n\n{sizes['n_items']:,} items, "
		f"{sizes['n_persons']:,} persons, {int(train.sum()):,} training cells. "
		f"Held-out log-likelihood {result.held_out_log_likelihood:.4f} per cell.\n\n"
		f"{body}\n\n{flags}\n",
		encoding="utf-8",
	)
	runmeta.write(
		out,
		runmeta.build(
			script="04_fit.py",
			config_raw=cfg.raw,
			extra={
				"effective": {**environment, "inference": args.loss},
				"dataset": sizes,
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
	)
	print(f"\nwrote {out}")


if __name__ == "__main__":
	main()
