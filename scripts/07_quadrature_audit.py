"""How the two quadrature rules err as one person's responses accumulate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from thesis import config, runmeta
from thesis.model import precision

RESPONSE_COUNTS = (1, 3, 10, 30, 100, 300, 1000)
NODE_COUNTS = (11, 21, 31, 41)
THETA_TRUE = 0.8


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--config", type=Path, default=None, help="path to config.toml")
	ap.add_argument("--precision", default=None, choices=precision.PRECISIONS)
	args = ap.parse_args()

	cfg = config.load(args.config)
	precision.enable(args.precision or cfg.model.precision)

	import jax
	import jax.numpy as jnp

	from thesis.eval import audit, synth
	from thesis.model import transform

	ranges = np.array(
		[cfg.synth.a, cfg.synth.b, cfg.synth.omega, cfg.synth.gamma_0, cfg.synth.gamma_1]
	)
	rng = np.random.default_rng(cfg.synth.seed)
	n_items = cfg.synth.ref.n_items
	z = synth.draw_z(rng, n_items, ranges)
	tau = transform.to_tau(jnp.asarray(z))

	modes = np.empty(len(RESPONSE_COUNTS))
	sds = np.empty(len(RESPONSE_COUNTS))
	standard = np.empty((len(RESPONSE_COUNTS), len(NODE_COUNTS)))
	adaptive = np.empty_like(standard)

	print(
		f"{'n_resp':>7} {'post sd':>8} "
		+ " ".join(f"{'Q=' + str(q):^19}" for q in NODE_COUNTS)
	)
	print(
		f"{'':>7} {'':>8} "
		+ " ".join(f"{'standard':>9} {'adaptive':>9}" for _ in NODE_COUNTS)
	)
	for row, n_resp in enumerate(RESPONSE_COUNTS):
		items = rng.integers(0, n_items, size=n_resp)
		ys = synth.draw_responses(
			rng, z, np.array([THETA_TRUE]), items, np.zeros(n_resp, dtype=int)
		)
		truth, modes[row], sds[row] = audit.exact(tau, items, ys)
		for col, nodes in enumerate(NODE_COUNTS):
			standard[row, col] = abs(
				audit.estimate(tau, items, ys, nodes, adaptive=False) - truth
			)
			adaptive[row, col] = abs(
				audit.estimate(tau, items, ys, nodes, adaptive=True) - truth
			)
		print(
			f"{n_resp:>7} {sds[row]:8.4f} "
			+ " ".join(
				f"{standard[row, c]:9.2e} {adaptive[row, c]:9.2e}"
				for c in range(len(NODE_COUNTS))
			)
		)

	cfg.ensure_dirs()
	out = cfg.paths.artifacts / "quadrature_audit.npz"
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
	runmeta.write(
		out,
		runmeta.build(
			script="07_quadrature_audit.py",
			config_raw=cfg.raw,
			extra={
				"effective": {
					"precision": args.precision or cfg.model.precision,
					"theta_true": THETA_TRUE,
					"devices": [str(d) for d in jax.devices()],
				},
				"worst_standard": float(standard.max()),
				"worst_adaptive": float(adaptive.max()),
			},
		),
	)
	print(f"\nwrote {out}")


if __name__ == "__main__":
	main()