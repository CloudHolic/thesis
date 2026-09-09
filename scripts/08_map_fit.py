"""ZOI Beta IRT's on real responses, by MAP Loss."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from thesis import config, domain, runmeta
from thesis.model import precision


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--config", type=Path, default=None, help="path to config.toml")
	ap.add_argument("--response", default=None, choices=sorted(domain.VIEWS))
	ap.add_argument("--pool", default=None, choices=list(domain.POOLS))
	ap.add_argument("--quadrature", type=int, default=None, help="overrides [model]")
	ap.add_argument("--precision", default=None, choices=precision.PRECISIONS)
	ap.add_argument("--steps", type=int, default=None, help="overrides [train]")
	args = ap.parse_args()

	cfg = config.load(args.config)
	precision.enable(args.precision or cfg.model.precision)

	import jax
	import jax.numpy as jnp

	from thesis.data import dataset
	from thesis.eval import recovery
	from thesis.model import encoder, marginal, quadrature, transform
	from thesis.train import loop, objective

	response_name = args.response or cfg.data.response
	pool = args.pool or cfg.data.pool
	nodes = args.quadrature or cfg.model.quadrature_nodes

	source = cfg.paths.artifacts / f"pilot_{response_name}_{pool}.npz"
	data, held_out = dataset.load(source)
	held = held_out.astype(bool)
	person = data.person_index()
	train = ~held
	print(
		f"data      {data.n_items:,} items, {data.n_persons:,} persons, "
		f"{data.n_obs:,} responses, {int(held.sum()):,} held out"
	)

	rule = quadrature.gauss_hermite(nodes)
	train_split = marginal.split_by_branch(
		data.item_index[train], person[train], data.response[train], data.n_persons
	)
	full_split = marginal.split_by_branch(data.item_index, person, data.response, data.n_persons)
	z0 = encoder.initial_z(data.response[train], data.n_items)

	def loss_fn(z: jnp.ndarray) -> jnp.ndarray:
		return objective.map_loss(z, rule, train_split)

	started = time.monotonic()
	fit = loop.run(
		loss_fn,
		z0,
		optimizer=loop.build(cfg.train.optimizer, cfg.train.learning_rate),
		steps=args.steps or cfg.train.steps,
		tolerance=cfg.train.tolerance * data.n_persons,
		patience=cfg.train.patience,
	)
	seconds = time.monotonic() - started
	step_seconds = float(np.median(fit.seconds[1:])) if fit.steps > 1 else float("nan")
	print(
		f"fit       {fit.steps} steps in {seconds:.1f}s, "
		f"then {1000 * step_seconds:.1f} ms/step, converged={fit.converged}"
	)

	tau = transform.to_tau(jnp.asarray(fit.z))
	# The conditional of the held-out cells given the training ones, each integral
	# centred on its own posterior.
	train_ll = float(marginal.log_marginal(tau, rule, train_split).sum())
	full_ll = float(marginal.log_marginal(tau, rule, full_split).sum())
	held_ll = (full_ll - train_ll) / int(held.sum())
	print(f"holdout   {held_ll:.4f} log-likelihood per cell")

	center = marginal.find_center(tau, train_split)
	usable = int(np.asarray(center.usable).sum())
	theta = np.asarray(center.mode)
	print(f"centres   {usable:,}/{data.n_persons:,} adaptive")

	quantities = recovery.quantities(recovery.tau_from_z(fit.z))
	print(f"\n{'quantity':>12} {'p05':>9} {'p25':>9} {'p50':>9} {'p75':>9} {'p95':>9}")
	for name, values in quantities.items():
		p = np.percentile(values, [5, 25, 50, 75, 95])
		print(f"{name:>12} " + " ".join(f"{v:9.3f}" for v in p))

	# Where b*(y) sits outside the theta range of the people who played the item, the
	# number is an extrapolation, not a measurement.
	lo = np.full(data.n_items, np.inf)
	hi = np.full(data.n_items, -np.inf)
	np.minimum.at(lo, data.item_index[train], theta[person[train]])
	np.maximum.at(hi, data.item_index[train], theta[person[train]])
	flags = {
		name: (values < lo) | (values > hi)
		for name, values in quantities.items()
		if name.startswith("b*")
	}
	print("\nextrapolated (b* outside the responders' theta range):")
	for name, flag in flags.items():
		print(f"  {name:>10} {int(flag.sum()):>6,} of {data.n_items:,}  ({float(flag.mean()):.1%})")

	cfg.ensure_dirs()
	out = cfg.paths.artifacts / f"id_fit_{response_name}_{pool}.npz"
	flag_names = sorted(flags)
	np.savez_compressed(
		out,
		fitted_z=fit.z,
		losses=fit.losses,
		theta_mode=theta,
		theta_usable=np.asarray(center.usable),
		item_theta_low=lo,
		item_theta_high=hi,
		flag_names=np.array(flag_names),
		extrapolated=np.stack([flags[name] for name in flag_names]),
	)
	runmeta.write(
		out,
		runmeta.build(
			script="08_id_fit.py",
			config_raw=cfg.raw,
			extra={
				"effective": {
					"response": response_name,
					"pool": pool,
					"quadrature_nodes": nodes,
					"precision": args.precision or cfg.model.precision,
					"devices": [str(d) for d in jax.devices()],
					"source": source.name,
				},
				"dataset": {
					"n_items": data.n_items,
					"n_persons": data.n_persons,
					"n_obs": data.n_obs,
					"held_out": int(held.sum()),
				},
				"fit": {
					"steps": fit.steps,
					"converged": fit.converged,
					"best_loss": fit.best_loss,
					"seconds": round(seconds, 1),
					"ms_per_step": round(1000 * step_seconds, 2),
					"adaptive_centres": usable,
				},
				"holdout_log_likelihood_per_cell": held_ll,
				"extrapolated": {name: float(flag.mean()) for name, flag in flags.items()},
			},
		),
	)
	print(f"\nwrote {out}")


if __name__ == "__main__":
	main()
