"""Verification: how the quadrature errs, and whether a fit recovers known truth.

uv run scripts/03_diagnose.py                        both
uv run scripts/03_diagnose.py --only quadrature      the audit alone
uv run scripts/03_diagnose.py --only recovery --fixture pilot
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from thesis import config, report, runmeta
from thesis.data import dataset
from thesis.diagnostics import audit, recovery, synth
from thesis.model import difficulty
from thesis.model import fit as model_fit
from thesis.model.family.registry import FAMILIES
from thesis.model.loss import zoi_map
from thesis.utils import precision

# Response counts and node counts the audit sweeps, and the theta it draws at.
RESPONSE_COUNTS = (1, 3, 10, 30, 100, 300, 1000)
NODE_COUNTS = (11, 21, 31, 41)
THETA_TRUE = 0.8

# True item parameters for the synthetic fixture, uniform in tau. Moment-matched to the
# observed acc marginal: E[y] = sigmoid(b), phi = 2 exp(omega/2) cosh(eta/2),
# P(y=1) = 1 - sigmoid(gamma_1). gamma_0 sits where the data cannot speak.
TAU_RANGES = np.array([[0.5, 2.5], [1.5, 5.0], [0.5, 3.5], [-7.0, -5.0], [2.5, 4.5]])
SYNTH_SEED = 20260907
REF_ITEMS = 12
REF_PERSONS = 200

STAGES = ("quadrature", "recovery")
FIXTURES = ("ref", "pilot")


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--config", type=Path, default=None, help="path to config.toml")
	ap.add_argument("--only", default=None, choices=STAGES, help="run one stage")
	ap.add_argument("--fixture", default="ref", choices=FIXTURES, help="recovery scale")
	ap.add_argument("--reference", action="store_true", help="also fit the paper's model")
	args = ap.parse_args()

	cfg = config.load(args.config)
	# Before any array exists, and therefore before anything else is touched.
	precision.enable(cfg.model.precision)
	family = FAMILIES["zoi_beta"]

	out_dir = cfg.directory("diagnostics")
	environment = {
		"precision": cfg.model.precision,
		"devices": [str(d) for d in jax.devices()],
	}
	stages = STAGES if args.only is None else (args.only,)

	if "quadrature" in stages:
		rng = np.random.default_rng(SYNTH_SEED)
		z = synth.draw_z(rng, REF_ITEMS, TAU_RANGES)
		tau = family.to_tau(jnp.asarray(z))

		modes = np.empty(len(RESPONSE_COUNTS))
		sds = np.empty(len(RESPONSE_COUNTS))
		standard = np.empty((len(RESPONSE_COUNTS), len(NODE_COUNTS)))
		adaptive = np.empty_like(standard)

		for row, n_resp in enumerate(RESPONSE_COUNTS):
			items = rng.integers(0, REF_ITEMS, size=n_resp)
			ys = synth.draw_responses(
				rng, z, np.array([THETA_TRUE]), items, np.zeros(n_resp, dtype=int)
			)
			truth, modes[row], sds[row] = audit.exact(family, tau, items, ys)
			for col, nodes in enumerate(NODE_COUNTS):
				standard[row, col] = abs(
					audit.estimate(family, tau, items, ys, nodes, adaptive=False) - truth
				)
				adaptive[row, col] = abs(
					audit.estimate(family, tau, items, ys, nodes, adaptive=True) - truth
				)

		header = (
			"| responses | posterior sd | "
			+ " | ".join(f"Q={q} standard | Q={q} adaptive" for q in NODE_COUNTS)
			+ " |"
		)
		table = [header, "|---:" * (2 + 2 * len(NODE_COUNTS)) + "|"]
		table += [
			f"| {n_resp} | {sds[row]:.4f} | "
			+ " | ".join(
				f"{standard[row, c]:.2e} | {adaptive[row, c]:.2e}" for c in range(len(NODE_COUNTS))
			)
			+ " |"
			for row, n_resp in enumerate(RESPONSE_COUNTS)
		]
		body = "\n".join(table)
		print(body)

		out = out_dir / "quadrature_audit.npz"
		np.savez_compressed(
			out,
			response_counts=np.array(RESPONSE_COUNTS),
			node_counts=np.array(NODE_COUNTS),
			posterior_mode=modes,
			posterior_sd=sds,
			standard_error=standard,
			adaptive_error=adaptive,
			true_z=z,
		)
		out.with_suffix(".md").write_text(
			f"# Quadrature audit\n\nAbsolute error in `log p(y_p)`, "
			f"precision `{cfg.model.precision}`.\n\n{body}\n",
			encoding="utf-8",
		)
		runmeta.write(
			out,
			runmeta.build(
				script="03_diagnose.py",
				config_raw=cfg.raw,
				extra={
					"effective": {**environment, "stage": "quadrature", "theta_true": THETA_TRUE},
					"worst_standard": float(standard.max()),
					"worst_adaptive": float(adaptive.max()),
				},
			),
		)
		print(f"\nwrote {out}")

	if "recovery" not in stages:
		return

	if args.fixture == "ref":
		n_items, n_persons = REF_ITEMS, REF_PERSONS
		item, person = synth.full_cross(n_items, n_persons)
	else:
		source = cfg.directory("dataset") / f"pilot_{cfg.data.pool}.npz"
		data = dataset.load(source)
		n_items, n_persons = data.n_items, data.n_persons
		item, person = synth.training_cells(data)

	fixture = synth.build(item, person, n_items, n_persons, TAU_RANGES, SYNTH_SEED)
	shape = {
		"at_zero": float((fixture.response == 0.0).mean()),
		"at_one": float((fixture.response == 1.0).mean()),
		"p50": float(np.median(fixture.response)),
	}
	print(f"\nfixture   {n_items:,} items, {n_persons:,} persons, {item.size:,} responses")
	print(f"response  {shape}")

	fit = model_fit.run(
		zoi_map,
		family=family,
		item_index=item,
		person_index=person,
		response=fixture.response,
		n_items=n_items,
		n_persons=n_persons,
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
	usable = int(fit.theta_usable.sum())
	print(
		f"map       {fit.steps} steps, first {fit.seconds[0]:.1f}s incl. compile, then "
		f"{1000 * fit.step_seconds:.1f} ms/step, converged={fit.converged}, "
		f"adaptive centres {usable:,}/{n_persons:,}"
	)

	truth = difficulty.quantities(family, family.to_tau(jnp.asarray(fixture.z)))
	tables = {"ours_vs_truth": recovery.compare(truth, fit.quantities)}
	blocks = [report.comparison(tables["ours_vs_truth"], "ours vs truth")]

	reference_meta: dict[str, object] = {}
	if args.reference:
		from thesis.references.zoi import mcmc

		started = time.monotonic()
		post = mcmc.run(
			item,
			person,
			fixture.response,
			n_items,
			n_persons,
			seed=cfg.train.seed,
			chains=cfg.reference.chains,
			samples=cfg.reference.samples,
			warmup=cfg.reference.warmup,
		)
		theirs = difficulty.quantities(family, family.tau_from_sites(post.mean))
		tables["reference_vs_truth"] = recovery.compare(truth, theirs)
		tables["ours_vs_reference"] = recovery.compare(theirs, fit.quantities)
		blocks += [
			report.comparison(tables["reference_vs_truth"], "paper vs truth"),
			report.comparison(tables["ours_vs_reference"], "ours vs paper"),
		]
		reference_meta = {
			"seconds": round(time.monotonic() - started, 1),
			"max_r_hat": max(float(np.nanmax(v)) for v in post.r_hat.values()),
			"divergences": post.divergences,
		}
		print(f"\nreference {reference_meta}")

	body = "\n\n".join(blocks)
	print(f"\n{body}")

	out = out_dir / f"recovery_{args.fixture}.npz"
	np.savez_compressed(
		out,
		true_z=fixture.z,
		fitted_z=fit.reported["z"],
		theta=fixture.theta,
		losses=fit.losses,
		item_index=item,
		person_index=person,
		response=fixture.response,
	)
	out.with_suffix(".md").write_text(
		f"# Synthetic recovery, `{args.fixture}` fixture\n\n"
		f"{n_items:,} items, {n_persons:,} persons, {item.size:,} responses. "
		f"Q={cfg.model.quadrature_nodes}, precision `{cfg.model.precision}`.\n\n{body}\n",
		encoding="utf-8",
	)
	runmeta.write(
		out,
		runmeta.build(
			script="03_diagnose.py",
			config_raw=cfg.raw,
			extra={
				"effective": {**environment, "stage": "recovery", "fixture": args.fixture},
				"dataset": {"n_items": n_items, "n_persons": n_persons, "n_obs": int(item.size)},
				"response_shape": shape,
				"fit": {
					"steps": fit.steps,
					"converged": fit.converged,
					"best_loss": fit.best_loss,
					"first_step_seconds": round(float(fit.seconds[0]), 2),
					"ms_per_step": round(1000 * fit.step_seconds, 2),
					"adaptive_centres": usable,
				},
				"reference": reference_meta,
				"recovery": tables,
			},
		),
	)
	print(f"\nwrote {out}")


if __name__ == "__main__":
	main()
