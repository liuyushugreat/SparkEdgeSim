"""CLI entry point for SparkEdgeSim — ``edge-unit-sim`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="edge-unit-sim",
    help="SparkEdgeSim — DGX & GP Spark Edge Compute Unit Simulator",
    add_completion=False,
)
console = Console()


@app.command()
def serve(
    config: Optional[Path] = typer.Option(  # noqa: UP007
        None, "--config", "-c", help="Path to YAML configuration file"
    ),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind address"),
    port: int = typer.Option(8080, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev mode)"),
) -> None:
    """Start the edge unit simulator as a REST API server."""
    import uvicorn

    from dgx_gp_spark_sim.api.server import create_app
    from dgx_gp_spark_sim.config import EdgeUnitConfig, load_config

    cfg = load_config(config) if config else EdgeUnitConfig()
    console.print(f"[bold green]Starting SparkEdgeSim[/] unit=[cyan]{cfg.unit_id}[/]")
    console.print(f"  Compute: {cfg.dgx_spark.gpu_tflops} TFLOPS | {cfg.dgx_spark.unified_memory_gb} GB unified memory")
    console.print(f"  Storage: {cfg.gp_spark.iops:,} IOPS | {cfg.gp_spark.storage_bandwidth_gbps} GB/s")
    console.print(f"  Network: {cfg.network.bandwidth_gbps} Gbps | RTT edge-edge {cfg.network.edge_edge_rtt_ms} ms")
    console.print(f"  Listening on [bold]{host}:{port}[/]")

    fastapi_app = create_app(cfg)
    uvicorn.run(fastapi_app, host=host, port=port, reload=reload)


@app.command()
def run_example(
    name: str = typer.Argument("single_task", help="Example name to run"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),  # noqa: UP007
) -> None:
    """Run a built-in example scenario."""
    import asyncio

    from dgx_gp_spark_sim.config import EdgeUnitConfig, load_config
    from dgx_gp_spark_sim.edge_unit.node import EdgeUnitNode
    from dgx_gp_spark_sim.models import BatchRequest, Task

    cfg = load_config(config) if config else EdgeUnitConfig()
    node = EdgeUnitNode(cfg)

    async def _single_task() -> None:
        task = Task(
            task_id="demo-001",
            op_type="risk_score",
            flops=1.2e8,
            input_bytes=4096,
            state_refs=["cell_12", "neighbor_window_12"],
        )
        await node.state.put("cell_12", {"risk": 0.3})
        await node.state.put("neighbor_window_12", {"positions": [1, 2, 3]})

        result = await node.submit_task(task)
        _print_result(result)

    async def _batch() -> None:
        tasks = [
            Task(task_id=f"batch-{i}", op_type="classify", flops=5e7, input_bytes=2048)
            for i in range(8)
        ]
        batch = BatchRequest(batch_id="demo-batch-001", tasks=tasks)
        result = await node.submit_batch(batch)
        console.print(f"\n[bold]Batch {result.batch_id}[/]: {result.tasks_completed} completed, "
                       f"total {result.total_latency_ms:.3f} ms")
        for r in result.results:
            _print_result(r)

    def _print_result(r: object) -> None:
        table = Table(title=f"Task {getattr(r, 'task_id', '?')}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        for field_name in ["status", "latency_ms", "compute_ms", "queue_delay_ms",
                           "state_access_ms", "transfer_ms", "state_hit_ratio"]:
            val = getattr(r, field_name, None)
            if val is not None:
                table.add_row(field_name, str(val))
        console.print(table)

    examples = {"single_task": _single_task, "batch": _batch}
    fn = examples.get(name)
    if fn is None:
        console.print(f"[red]Unknown example:[/] {name}")
        console.print(f"Available: {', '.join(examples.keys())}")
        raise typer.Exit(1)

    console.print(f"[bold green]Running example:[/] {name}")
    asyncio.run(fn())

    console.print("\n[bold]Metrics:[/]")
    import json

    console.print_json(json.dumps(node.get_metrics(), indent=2))


@app.command()
def benchmark(
    tasks: int = typer.Option(1000, "--tasks", "-n", help="Number of tasks"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size (1 = no batching)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c"),  # noqa: UP007
) -> None:
    """Run a simple throughput benchmark."""
    import asyncio
    import time

    from dgx_gp_spark_sim.config import EdgeUnitConfig, load_config
    from dgx_gp_spark_sim.edge_unit.node import EdgeUnitNode
    from dgx_gp_spark_sim.models import BatchRequest, Task

    cfg = load_config(config) if config else EdgeUnitConfig()
    node = EdgeUnitNode(cfg)

    async def _run() -> None:
        t0 = time.monotonic()

        if batch_size <= 1:
            for i in range(tasks):
                task = Task(task_id=f"bench-{i}", flops=1e7, input_bytes=1024)
                await node.submit_task(task)
        else:
            for start in range(0, tasks, batch_size):
                end = min(start + batch_size, tasks)
                batch_tasks = [
                    Task(task_id=f"bench-{i}", flops=1e7, input_bytes=1024)
                    for i in range(start, end)
                ]
                batch = BatchRequest(batch_id=f"bench-batch-{start}", tasks=batch_tasks)
                await node.submit_batch(batch)

        elapsed = time.monotonic() - t0
        console.print(f"\n[bold green]Benchmark complete[/]")
        console.print(f"  Tasks: {tasks}")
        console.print(f"  Batch size: {batch_size}")
        console.print(f"  Elapsed: {elapsed:.3f} s")
        console.print(f"  Throughput: {tasks / elapsed:.1f} tasks/s")

    asyncio.run(_run())

    import json

    console.print("\n[bold]Metrics:[/]")
    console.print_json(json.dumps(node.get_metrics(), indent=2))


if __name__ == "__main__":
    app()
