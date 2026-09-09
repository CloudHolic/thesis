"""ZOI Beta IRT's on real responses, by NUTS."""

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
	ap.add_argument("--precision", default=None, choices=precision.PRECISIONS)
	ap.add_argument("--chains", type=int, default=None, help="overrides [synth.ref]")
	ap.add_argument("--samples", type=int, default=None, help="overrides [synth.ref]")
	ap.add_argument("--warmup", type=int, default=None, help="overrides [synth.ref]")
	ap.add_argument("--label", default=None, help="suffixes the artifact name")
	args = ap.parse_args()

	cfg = config.load(args.config)
	precision.enable(args.precision or cfg.model.precision)

	import jax

	from thesis.data import dataset
	from thesis.eval import reference

	response_name = args.response or cfg.data.response
	pool = args.pool or cfg.data.pool
	chains = args.chains or cfg.synth.ref.chains
	samples = args.samples or cfg.synth.ref.samples
	warmup = args.warmup or cfg.synth.ref.warmup

	source = cfg.paths.artifacts / f"pilot_{response_name}_{pool}.npz"
	data, held_out = dataset.load(source)
	train = ~held_out.astype(bool)
	person = data.person_index()[train]
	item = data.item_index[train]
	y = data.response[train]
	latent = data.n_persons + 5 * data.n_items
	print(
		f"data      {data.n_items:,} items, {data.n_persons:,} persons, "
		f"{item.size:,} training cells, {latent:,} latent parameters"
	)
	print(f"nuts      {chains} chains x {samples} samples after {warmup} warmup")

	started = time.monotonic()
	post = reference.run(
		item,
		person,
		y,
		data.n_items,
		data.n_persons,
		seed=cfg.train.seed,
		chains=chains,
		samples=samples,
		warmup=warmup,
	)
	seconds = time.monotonic() - started
	worst = {site: float(np.nanmax(v)) for site, v in post.r_hat.items()}
	print(
		f"done      {seconds / 60:.1f} min, divergences {post.divergences:,}, "
		f"max r_hat {max(worst.values()):.4f}"
	)
	for site, value in worst.items():
		print(f"  {site:>9} r_hat {value:.4f}")

	cfg.ensure_dirs()
	suffix = f"_{args.label}" if args.label else ""
	out = cfg.paths.artifacts / f"reference_{response_name}_{pool}{suffix}.npz"

	sites = list(reference.SITES)
	np.savez_compressed(
		out,
		sites=np.array(sites),
		mean=np.stack([post.mean[s] for s in sites]),
		median=np.stack([post.median[s] for s in sites]),
		lower=np.stack([post.lower[s] for s in sites]),
		upper=np.stack([post.upper[s] for s in sites]),
		r_hat=np.stack([post.r_hat[s] for s in sites]),
	)

	runmeta.write(
		out,
		runmeta.build(
			script="08_reference_fit.py",
			config_raw=cfg.raw,
			extra={
				"effective": {
					"response": response_name,
					"pool": pool,
					"precision": args.precision or cfg.model.precision,
					"chains": chains,
					"samples": samples,
					"warmup": warmup,
					"devices": [str(d) for d in jax.devices()],
					"source": source.name,
				},
				"dataset": {
					"n_items": data.n_items,
					"n_persons": data.n_persons,
					"n_train": int(item.size),
					"latent_parameters": latent,
				},
				"nuts": {
					"minutes": round(seconds / 60, 1),
					"divergences": post.divergences,
					"max_r_hat": max(worst.values()),
					"r_hat_by_site": worst,
				},
			},
		),
	)
	print(f"\nwrote {out}")


if __name__ == "__main__":
	main()
