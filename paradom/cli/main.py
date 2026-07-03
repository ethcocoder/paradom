import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Optional
from pathlib import Path
import json

app = typer.Typer(help="Paradom Framework CLI")
console = Console()

@app.command()
def swap(
    source: str = typer.Argument(..., help="Path to source model or HF ID"),
    target_arch: str = typer.Option(..., "--target-arch", help="Target architecture name"),
    config: str = typer.Option("{}", "--config", help="Path to target config JSON/YAML"),
    output: str = typer.Option("./output", "--output", help="Output directory"),
    fraction: float = typer.Option(0.35, "--fraction", help="Swap fraction (importance-based)"),
    source_arch: str = typer.Option("tinytransformer", "--source-arch", help="Source architecture"),
):
    """Swap weights from source model to target architecture."""
    console.print(Panel(
        f"Starting weight swap: [bold blue]{source}[/bold blue] → [bold green]{target_arch}[/bold green]",
        title="Paradom Swap",
    ))

    from paradom.core.engine import Paradom
    engine = Paradom()

    try:
        report = engine.swap(
            source=source,
            target_architecture=target_arch,
            target_config={},
            source_architecture=source_arch,
            swap_fraction=fraction,
            output_path=output,
        )

        console.print("\n[bold green]Swap complete![/bold green]")
        console.print(f"Weights swapped: {report.weights_swapped} / {report.total_weights}")
        console.print(f"Mean CKA: {report.mean_cka:.3f}")
        console.print(f"Quality Tier: [bold cyan]{report.quality_tier.value}[/bold cyan]")
        console.print(f"Time: {report.conversion_time_seconds:.2f}s")
        console.print(f"Output: {output}/model.safetensors")

        report_path = Path(output) / "swap_report.json"
        if not report_path.exists():
            with open(report_path, "w") as f:
                report_dict = {
                    k: (v.value if hasattr(v, "value") else v)
                    for k, v in report.__dict__.items()
                }
                json.dump(report_dict, f, indent=2, default=str)

    except Exception as e:
        console.print(f"[bold red]Error during swap:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def identify(
    source: str = typer.Argument(..., help="Path to source model checkpoint"),
    target_arch: str = typer.Option("tinymamba", "--target-arch", help="Target architecture name"),
    source_arch: str = typer.Option("tinytransformer", "--source-arch", help="Source architecture"),
    output: Optional[str] = typer.Option(None, "--output", help="Save JSON equivalence map"),
):
    """Identify weight equivalences between models."""
    console.print(Panel(
        f"Identifying equivalences: [bold blue]{source}[/bold blue] → [bold green]{target_arch}[/bold green]",
        title="Paradom Identify",
    ))

    from paradom.core.engine import Paradom
    engine = Paradom()
    eq_map = engine.identify(source, target_arch, {}, source_arch)

    table = Table(title="Equivalence Map")
    table.add_column("Source", style="cyan")
    table.add_column("Target", style="green")
    table.add_column("Type")
    table.add_column("CKA", justify="right")

    for pair in eq_map.pairs:
        table.add_row(
            pair.source.name,
            pair.target_layer_name,
            pair.swap_type.value,
            f"{pair.cka_score:.3f}",
        )

    console.print(table)
    console.print(f"\nMean CKA: {eq_map.mean_cka:.3f}")
    console.print(f"Estimated tier: {eq_map.estimated_quality_tier.value}")
    console.print(f"Unmapped source keys: {len(eq_map.unmapped_source)}")

    if output:
        payload = {
            "mean_cka": eq_map.mean_cka,
            "quality_tier": eq_map.estimated_quality_tier.value,
            "pairs": [
                {
                    "source": p.source.name,
                    "target": p.target_layer_name,
                    "swap_type": p.swap_type.value,
                    "cka_score": p.cka_score,
                }
                for p in eq_map.pairs
            ],
            "unmapped_source": eq_map.unmapped_source,
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(payload, f, indent=2)
        console.print(f"Saved to {output}")


if __name__ == "__main__":
    app()
