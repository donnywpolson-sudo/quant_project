from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


class DatabentoMetadataClient(Protocol):
    def get_cost(self, **kwargs: object) -> float: ...

    def get_billable_size(self, **kwargs: object) -> object: ...


class DatabentoClient(Protocol):
    metadata: DatabentoMetadataClient


def _relative_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_plan(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.exists():
        return None, [f"missing tick/source gap plan: {_relative_path(path)}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"unreadable tick/source gap plan: {exc}"]
    if not isinstance(payload, dict):
        return None, ["tick/source gap plan top-level JSON is not an object"]
    return payload, []


def _cost_request_from_task(task: dict[str, Any], source_plan: Path) -> dict[str, Any]:
    return {
        "source_plan": _relative_path(source_plan),
        "market": task.get("market"),
        "year": task.get("year"),
        "dataset": task.get("dataset"),
        "symbols": str(task.get("instrument_id")),
        "schema": task.get("schema"),
        "stype_in": task.get("stype_in"),
        "start": task.get("start"),
        "end": task.get("end"),
        "reason": task.get("reason"),
        "source_gap_timestamps": task.get("source_gap_timestamps"),
        "raw_ohlcv_source_file": task.get("raw_ohlcv_source_file"),
        "raw_ohlcv_source_hash": task.get("raw_ohlcv_source_hash"),
    }


def build_cost_request_plan(plan_paths: list[Path]) -> dict[str, Any]:
    failures: list[str] = []
    requests: list[dict[str, Any]] = []
    for path in plan_paths:
        plan, load_failures = _load_plan(path)
        failures.extend(load_failures)
        if plan is None:
            continue
        if plan.get("status") != "PASS":
            failures.append(f"source plan is not PASS: {_relative_path(path)}")
        tasks = plan.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            failures.append(f"source plan has no tasks: {_relative_path(path)}")
            continue
        for task in tasks:
            if not isinstance(task, dict):
                failures.append(f"source plan has non-object task: {_relative_path(path)}")
                continue
            requests.append(_cost_request_from_task(task, path))

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "FAIL" if failures else "PASS",
        "estimate_only": True,
        "download_allowed": False,
        "api_called": False,
        "failures": failures,
        "request_count": len(requests),
        "requests": requests,
    }


def estimate_costs(
    request_plan: dict[str, Any],
    client: DatabentoClient,
) -> dict[str, Any]:
    failures = list(request_plan.get("failures", []))
    estimates: list[dict[str, Any]] = []
    if request_plan.get("status") != "PASS":
        return {
            **request_plan,
            "api_called": False,
            "status": "FAIL",
            "failures": failures,
            "estimates": estimates,
            "total_estimated_cost_usd": None,
        }

    total_cost = 0.0
    for request in request_plan.get("requests", []):
        kwargs = {
            "dataset": request.get("dataset"),
            "symbols": request.get("symbols"),
            "schema": request.get("schema"),
            "stype_in": request.get("stype_in"),
            "start": request.get("start"),
            "end": request.get("end"),
        }
        try:
            cost = float(client.metadata.get_cost(**kwargs))
            size = client.metadata.get_billable_size(**kwargs)
        except Exception as exc:
            failures.append(
                f"{request.get('market')} {request.get('year')} {request.get('schema')}: {exc}"
            )
            estimates.append({**request, "status": "estimate_error", "error": str(exc)})
            continue
        total_cost += cost
        estimates.append(
            {
                **request,
                "status": "ok",
                "estimated_cost_usd": cost,
                "billable_size": size,
            }
        )

    return {
        **request_plan,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "api_called": True,
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "estimates": estimates,
        "total_estimated_cost_usd": total_cost if not failures else None,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", nargs="+", required=True)
    parser.add_argument("--estimate-out", required=True)
    parser.add_argument("--estimate-cost", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    request_plan = build_cost_request_plan([Path(path) for path in args.plan_json])
    output = request_plan
    if args.estimate_cost:
        if not args.allow_network:
            output = {
                **request_plan,
                "status": "FAIL",
                "failures": [
                    *request_plan.get("failures", []),
                    "--estimate-cost requires --allow-network",
                ],
            }
        else:
            from scripts.phase1A_download.download_databento_raw import get_client

            output = estimate_costs(request_plan, get_client())
    write_json(output, Path(args.estimate_out))
    if output["status"] != "PASS":
        print(f"FAIL tick/source cost estimate plan: failures={len(output['failures'])}")
        return 1
    mode = "estimate" if output.get("api_called") else "request"
    print(
        f"PASS tick/source cost {mode}: requests={output.get('request_count', len(output.get('requests', [])))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
