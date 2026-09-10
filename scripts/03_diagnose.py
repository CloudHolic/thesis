"""Verification: how the quadrature errs, and whether a fit recovers known truth.

uv run scripts/03_diagnose.py                        both
uv run scripts/03_diagnose.py --only quadrature      the audit alone
uv run scripts/03_diagnose.py --only recovery --fixture pilot
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from thesis import artifact, config, report
from thesis.data import dataset
from thesis.diagnostics import audit, cells, compare
from thesis.diagnostics.cells import Fixture
from thesis.diagnostics.generate.protocol import Generator
from thesis.diagnostics.generate.registry import GENERATORS
from thesis.model import difficulty
from thesis.model import fit as model_fit
from thesis.model.family.protocol import Family
from thesis.model.family.registry import FAMILIES
from thesis.model.loss import map as map_loss
from thesis.utils import precision

SCRIPT = Path(__file__).name

# Response counts and node counts the audit sweeps, and the theta it draws at.
RESPONSE_COUNTS = (1, 3, 10, 30, 100, 300, 1000)
NODE_COUNTS = (11, 21, 31, 41)
THETA_TRUE = 0.8

REF_ITEMS = 12
REF_PERSONS = 200

FAMILY_NAMES = tuple(sorted(FAMILIES))
STAGES = ("quadrature", "recovery")
FIXTURES = ("ref", "pilot")


def environment(cfg: config.Config, args: argparse.Namespace) -> dict[str, Any]:
	"""What the provenance record says about where this run happened."""
	return {
		"family": args.family,
		"precision": cfg.model.precision,
		"devices": [str(d) for d in jax.devices()],
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


def audit_table(sds: np.ndarray, standard: np.ndarray, adaptive: np.ndarray) -> str:
	"""Absolute error per response count, standard beside adaptive at each node count."""
	header = (
		"| responses | posterior sd | "
		+ " | ".join(f"Q={q} standard | Q={q} adaptive" for q in NODE_COUNTS)
		+ " |"
	)
	rows = [header, "|---:" * (2 + 2 * len(NODE_COUNTS)) + "|"]
	rows += [
		f"| {n_resp} | {sds[row]:.4f} | "
		+ " | ".join(
			f"{standard[row, c]:.2e} | {adaptive[row, c]:.2e}" for c in range(len(NODE_COUNTS))
		)
		+ " |"
		for row, n_resp in enumerate(RESPONSE_COUNTS)
	]

	return "\n".join(rows)


def run_quadrature(
	cfg: config.Config, args: argparse.Namespace, family: Family, generator: Generator
) -> None:
	"""Sweep response counts against node counts, and record how far each estimate falls."""
	rng = np.random.default_rng(generator.SEED)
	z = generator.draw_z(rng, REF_ITEMS, generator.TAU_RANGES)
	tau = family.to_tau(jnp.asarray(z))

	modes = np.empty(len(RESPONSE_COUNTS))
	sds = np.empty(len(RESPONSE_COUNTS))
	standard = np.empty((len(RESPONSE_COUNTS), len(NODE_COUNTS)))
	adaptive = np.empty_like(standard)

	for row, n_resp in enumerate(RESPONSE_COUNTS):
		items = rng.integers(0, REF_ITEMS, size=n_resp)
		ys = generator.draw_responses(
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

	body = audit_table(sds, standard, adaptive)
	print(body)

	out = cfg.directory("diagnostics") / f"quadrature_audit_{args.family}.npz"
	artifact.write(
		out,
		artifact.Artifact(
			arrays={
				"response_counts": np.array(RESPONSE_COUNTS),
				"node_counts": np.array(NODE_COUNTS),
				"posterior_mode": modes,
				"posterior_sd": sds,
				"standard_error": standard,
				"adaptive_error": adaptive,
				"true_z": z,
			},
			title="Quadrature audit",
			preamble=f"Absolute error in `log p(y_p)`, precision `{cfg.model.precision}`.",
			body=body,
			extra={
				"effective": {
					**environment(cfg, args),
					"stage": "quadrature",
					"theta_true": THETA_TRUE,
				},
				"worst_standard": float(standard.max()),
				"worst_adaptive": float(adaptive.max()),
			},
		),
		script=SCRIPT,
		config_raw=cfg.raw,
	)
	print(f"\nwrote {out}")


def build_fixture(cfg: config.Config, args: argparse.Namespace, generator: Generator) -> Fixture:
	"""Draw synthetic responses, on the reference grid or on the cells the pilot fit trains on."""
	if args.fixture == "ref":
		n_items, n_persons = REF_ITEMS, REF_PERSONS
		item, person = cells.full_cross(n_items, n_persons)
	else:
		source = cfg.directory("dataset") / f"pilot_{cfg.data.pool}.npz"
		data = dataset.load(source)
		n_items, n_persons = data.n_items, data.n_persons
		item, person = cells.training_cells(data)

	return generator.build(item, person, n_items, n_persons, generator.TAU_RANGES, generator.SEED)


def compare_with_reference(
	cfg: config.Config,
	family: Family,
	fixture: Fixture,
	truth: dict[str, np.ndarray],
	ours: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, dict[str, float]]], list[str], dict[str, Any]]:
	"""The paper's posterior on the same fixture, against truth and against ours."""
	# Importing numpyro is expensive, and a run without --reference never needs it.
	from thesis.references.zoi import mcmc

	started = time.monotonic()
	post = mcmc.run(
		fixture.item_index,
		fixture.person_index,
		fixture.response,
		fixture.n_items,
		fixture.n_persons,
		seed=cfg.train.seed,
		chains=cfg.reference.chains,
		samples=cfg.reference.samples,
		warmup=cfg.reference.warmup,
	)
	theirs = difficulty.quantities(family, family.tau_from_sites(post.mean))
	tables = {
		"reference_vs_truth": compare.compare(truth, theirs),
		"ours_vs_reference": compare.compare(theirs, ours),
	}
	blocks = [
		report.comparison(tables["reference_vs_truth"], "paper vs truth"),
		report.comparison(tables["ours_vs_reference"], "ours vs paper"),
	]
	meta = {
		"seconds": round(time.monotonic() - started, 1),
		"max_r_hat": max(float(np.nanmax(v)) for v in post.r_hat.values()),
		"divergences": post.divergences,
	}

	return tables, blocks, meta


def run_recovery(
	cfg: config.Config, args: argparse.Namespace, family: Family, generator: Generator
) -> None:
	"""Fit a fixture whose truth we drew, and report how much of that truth comes back."""
	fixture = build_fixture(cfg, args, generator)
	shape = {
		"at_zero": float((fixture.response == 0.0).mean()),
		"at_one": float((fixture.response == 1.0).mean()),
		"p50": float(np.median(fixture.response)),
	}
	print(
		f"\nfixture   {fixture.n_items:,} items, {fixture.n_persons:,} persons, "
		f"{fixture.item_index.size:,} responses"
	)
	print(f"response  {shape}")

	fit = model_fit.run(
		map_loss.build(family),
		family=family,
		item_index=fixture.item_index,
		person_index=fixture.person_index,
		response=fixture.response,
		n_items=fixture.n_items,
		n_persons=fixture.n_persons,
		settings=settings(cfg),
	)
	usable = int(fit.theta_usable.sum())
	print(
		f"map       {fit.steps} steps, first {fit.seconds[0]:.1f}s incl. compile, then "
		f"{1000 * fit.step_seconds:.1f} ms/step, converged={fit.converged}, "
		f"adaptive centres {usable:,}/{fixture.n_persons:,}"
	)

	truth = difficulty.quantities(family, family.to_tau(jnp.asarray(fixture.z)))
	tables: dict[str, Any] = {"ours_vs_truth": compare.compare(truth, fit.quantities)}
	blocks = [report.comparison(tables["ours_vs_truth"], "ours vs truth")]

	reference_meta: dict[str, Any] = {}
	if args.reference:
		more_tables, more_blocks, reference_meta = compare_with_reference(
			cfg, family, fixture, truth, fit.quantities
		)
		tables.update(more_tables)
		blocks += more_blocks
		print(f"\nreference {reference_meta}")

	body = "\n\n".join(blocks)
	print(f"\n{body}")

	out = cfg.directory("diagnostics") / f"recovery_{args.family}_{args.fixture}.npz"
	artifact.write(
		out,
		artifact.Artifact(
			arrays={
				"true_z": fixture.z,
				"fitted_z": fit.reported["z"],
				"theta": fixture.theta,
				"losses": fit.losses,
				"item_index": fixture.item_index,
				"person_index": fixture.person_index,
				"response": fixture.response,
			},
			title=f"Synthetic recovery, `{args.fixture}` fixture",
			preamble=(
				f"{fixture.n_items:,} items, {fixture.n_persons:,} persons, "
				f"{fixture.item_index.size:,} responses. "
				f"Q={cfg.model.quadrature_nodes}, precision `{cfg.model.precision}`."
			),
			body=body,
			extra={
				"effective": {
					**environment(cfg, args),
					"stage": "recovery",
					"fixture": args.fixture,
				},
				"dataset": {
					"n_items": fixture.n_items,
					"n_persons": fixture.n_persons,
					"n_obs": int(fixture.item_index.size),
				},
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
		script=SCRIPT,
		config_raw=cfg.raw,
	)
	print(f"\nwrote {out}")


def parse_args() -> argparse.Namespace:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--config", type=Path, default=None, help="path to config.toml")
	ap.add_argument("--only", default=None, choices=STAGES, help="run one stage")
	ap.add_argument("--fixture", default="ref", choices=FIXTURES, help="recovery scale")
	ap.add_argument("--family", default="zoi_beta", choices=FAMILY_NAMES)
	ap.add_argument("--reference", action="store_true", help="also fit the paper's model")

	return ap.parse_args()


def main() -> None:
	args = parse_args()
	cfg = config.load(args.config)
	# Before any array exists, and therefore before anything else is touched.
	precision.enable(cfg.model.precision)

	family = FAMILIES[args.family]
	generator = GENERATORS[args.family]
	stages = STAGES if args.only is None else (args.only,)

	if "quadrature" in stages:
		run_quadrature(cfg, args, family, generator)
	if "recovery" in stages:
		run_recovery(cfg, args, family, generator)


if __name__ == "__main__":
	main()
