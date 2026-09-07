"""Synthetic recovery: fit data generated from known truth, and report what comes back."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from thesis import config, runmeta
from thesis.model import precision

FIXTURES = ("ref", "pilot")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=None, help="path to config.toml")
    ap.add_argument("--fixture", default="ref", choices=FIXTURES)
    ap.add_argument("--quadrature", type=int, default=None, help="overrides [model]")
    ap.add_argument("--precision", default=None, choices=precision.PRECISIONS)
    ap.add_argument("--reference", action="store_true", help="also fit the paper's model")
    args = ap.parse_args()

    cfg = config.load(args.config)
    # Before any array exists, and therefore before anything else is touched.
    precision.enable(args.precision or cfg.model.precision)

    import jax

    from thesis.eval import recovery, synth
    from thesis.model import encoder, marginal, quadrature, transform
    from thesis.train import loop, objective

    nodes = args.quadrature or cfg.model.quadrature_nodes
    ranges = np.array([cfg.synth.a, cfg.synth.b, cfg.synth.omega, cfg.synth.gamma_0, cfg.synth.gamma_1])

    if args.fixture == "ref":
        n_items, n_persons = cfg.synth.ref.n_items, cfg.synth.ref.n_persons
        item, person = synth.full_cross(n_items, n_persons)
    else:
        source = cfg.paths.artifacts / f"pilot_{cfg.data.response}_{cfg.data.pool}.npz"
        item, person, n_items, n_persons = synth.dataset_cells(source)

    fixture = synth.build(item, person, n_items, n_persons, ranges, cfg.synth.seed)
    shape = {
        "at_zero": float((fixture.response == 0.0).mean()),
        "at_one": float((fixture.response == 1.0).mean()),
        "p50": float(np.median(fixture.response)),
    }

    print(f"fixture   {n_items:,} items, {n_persons:,} persons, {item.size:,} responses")
    print(f"response  {shape}")

    split = marginal.split_by_branch(item, person, fixture.response, n_persons)
    rule = quadrature.gauss_hermite(nodes)
    started = time.monotonic()
    fit = loop.run(
        lambda z: objective.map_loss(z, rule, split),
        encoder.initial_z(fixture.response, n_items),
        optimizer=loop.build(cfg.train.optimizer, cfg.train.learning_rate),
        steps=cfg.train.steps,
        tolerance=cfg.train.tolerance * n_persons,
        patience=cfg.train.patience
    )
    map_seconds = time.monotonic() - started
    center = marginal.find_center(transform.to_tau(np.asarray(fit.z)), split)
    usable = int(np.asarray(center.usable).sum())

    print(
        f"map       {fit.steps} steps in {map_seconds:.1f}s "
        f"({1000 * map_seconds / fit.steps:.1f} ms/step), converged={fit.converged}, "
        f"adaptive centres {usable:,}/{n_persons:,}"
    )

    truth = recovery.quantities(recovery.tau_from_z(fixture.z))
    ours = recovery.quantities(recovery.tau_from_z(fit.z))
    tables = {"ours_vs_truth": recovery.compare(truth, ours)}
    print("\n" + recovery.render(tables["ours_vs_truth"], "ours/truth"))

    reference_meta: dict[str, object] = {}
    if args.reference:
        from thesis.eval import reference

        started = time.monotonic()
        post = reference.run(
            item, person, fixture.response, n_items, n_persons,
            seed=cfg.train.seed,
            chains=cfg.synth.ref.chains,
            samples=cfg.synth.ref.samples,
            warmup=cfg.synth.ref.warmup
        )
        theirs = recovery.quantities(recovery.tau_from_posterior(post))
        tables["reference_vs_truth"] = recovery.compare(truth, theirs)
        tables["ours_vs_reference"] = recovery.compare(theirs, ours)
        reference_meta = {
            "seconds": round(time.monotonic() - started, 1),
            "max_r_hat": max(float(np.nanmax(v)) for v in post.r_hat.values()),
            "divergences": post.divergences
        }

        print(f"\nreference {reference_meta}")
        print("\n" + recovery.render(tables["reference_vs_truth"], "paper/truth"))
        print("\n" + recovery.render(tables["ours_vs_reference"], "ours/paper"))

    cfg.ensure_dirs()
    out = cfg.paths.artifacts / f"recovery_{args.fixture}.npz"
    np.savez_compressed(
        out,
        true_z=fixture.z,
        fitted_z=fit.z,
        theta=fixture.theta,
        losses=fit.losses,
        item_index=item,
        person_index=person,
        response=fixture.response
    )

    runmeta.write(
        out,
        runmeta.build(
            script="07_synthetic_recovery.py",
            config_raw=cfg.raw,
            extra={
                "effective": {
                    "fixture": args.fixture,
                    "quadrature_nodes": nodes,
                    "precision": args.precision or cfg.model.precision,
                    "devices": [str(d) for d in jax.devices()]
                },
                "dataset": {
                    "n_items": n_items, "n_persons": n_persons, "n_obs": int(item.size)
                },
                "response_shape": shape,
                "fit": {
                    "steps": fit.steps,
                    "converged": fit.converged,
                    "best_loss": fit.best_loss,
                    "seconds": round(map_seconds, 1),
                    "ms_per_step": round(1000 * map_seconds / fit.steps, 2),
                    "adaptive_centres": usable
                },
                "reference": reference_meta,
                "recovery": tables
            }
        )
    )

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()