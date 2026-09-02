"""Installed console-script adapters for modules with existing CLIs."""

import runpy


def _run(module: str) -> None:
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def dm_search() -> None:
    _run("scripts.errors_params")


def find_dm_params() -> None:
    _run("scripts.find_best_params")


def synthetic_benchmark() -> None:
    _run("scripts.sim_algo_check")


def trap_distributions() -> None:
    _run("scripts.trap_distr_builder")


def stationarity() -> None:
    _run("scripts.stationarity")
