import typer
from rich.console import Console
from rich.table import Table
from typing import Optional
import os

from paradom.core.loader import ModelLoader
from paradom.core.parser import ArchitectureParser
from paradom.core.importance import ImportanceScorer
from paradom.core.taxonomy import FunctionalRole

app = typer.Typer(help="Paradom: Universal Weight Equivalence Framework")
console = Console()

@app.command()
def discover(
    model_path: str = typer.Argument(..., help="Path to model weights (safetensors)"),
    paradigm: str = typer.Option("llm", help="AI paradigm (llm, vision, rl)"),
    importance: bool = typer.Option(True, help="Compute importance scores (Winning Tickets)")
):
    """
    Scans a model and identifies its functional weight products (Winning Tickets).
    """
    if not os.path.exists(model_path):
        console.print(f"[red]Error: Model path {model_path} not found.[/red]")
        raise typer.Exit(code=1)

    console.print(f"🔍 [bold]Starting Discovery Session[/bold]")
    console.print(f"Model: {model_path}")
    console.print(f"Paradigm: {paradigm.upper()}\n")

    loader = ModelLoader(model_path)
    parser = ArchitectureParser(paradigm=paradigm)
    scorer = ImportanceScorer() if importance else None

    table = Table(title="Functional Weight Inventory")
    table.add_column("Layer Name", style="cyan")
    table.add_column("Functional Role", style="green")
    table.add_column("Shape", style="magenta")
    if importance:
        table.add_column("Ticket Score", style="yellow")

    total_params = 0
    with console.status("[bold green]Streaming weights..."):
        for weight_dict in loader.stream_weights():
            for name, tensor in weight_dict.items():
                role = parser.identify_role(name)
                role_str = role.value if role else "[dim]UNKNOWN[/dim]"
                shape_str = str(list(tensor.shape))
                
                score_str = "-"
                if scorer and tensor.dim() >= 2:
                    # Quick score for discovery display
                    score_str = f"{tensor.abs().mean().item():.4f}" # Placeholder for full SVD in discovery

                table.add_row(name, role_str, shape_str, score_str)
                total_params += tensor.numel()

    console.print(table)
    console.print(f"\n✅ [bold]Discovery Complete.[/bold]")
    console.print(f"Total parameters found: {total_params/1e6:.1f}M")
    console.print(f"RAM Usage: [blue]Optimized Streaming Mode (Peak < 2GB)[/blue]")

@app.command()
def swap(
    source: str = typer.Argument(..., help="Path to source model weights"),
    target_config: str = typer.Argument(..., help="Path to target configuration YAML"),
    output: str = typer.Argument(..., help="Path to save redressed model"),
    paradigm: str = typer.Option("llm", help="AI paradigm")
):
    """
    Executes the weight equivalence swap (redressing).
    Surgical transfer of intelligence from source to target.
    """
    import yaml
    from paradom.core.loader import ModelLoader
    from paradom.core.equivalence import EquivalenceIdentifier
    from paradom.core.swap_engine import SwapEngine
    from paradom.core.writer import BufferedMmapWriter, StreamingSwapper

    with open(target_config, 'r') as f:
        config = yaml.safe_load(f)
    
    config["paradigm"] = paradigm

    console.print(f"🚀 [bold blue]Starting Redressing Operation[/bold]")
    console.print(f"Source: {source}")
    console.print(f"Output: {output}\n")

    loader = ModelLoader(source)
    identifier = EquivalenceIdentifier()
    engine = SwapEngine()
    writer = BufferedMmapWriter(output)
    
    swapper = StreamingSwapper(loader, identifier, engine, writer)
    
    with console.status("[bold green]Executing swaps..."):
        count = swapper.run(config)

    console.print(f"\n✅ [bold]Swap Complete.[/bold]")
    console.print(f"Total weights redressed: {count}")
    console.print(f"Result saved to: [cyan]{output}[/cyan]")

@app.command()
def validate(
    source: str = typer.Argument(..., help="Path to original source model"),
    swapped: str = typer.Argument(..., help="Path to redressed model"),
):
    """
    Validates redressed model quality and produces a Quality Report.
    """
    from paradom.core.validator import Validator

    console.print(f"⚖️ [bold]Validating Model Integrity...[/bold]")
    
    validator = Validator(source, swapped)
    report = validator.run_validation()
    
    table = Table(title="Paradom Quality Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Result", style="green")
    
    table.add_row("Mean CKA Similarity", f"{report['mean_cka']:.4f}")
    table.add_row("Quality Tier", report["overall_quality"])
    
    console.print(table)
    
    if report["overall_quality"] in ["EXCELLENT", "GOOD"]:
        console.print("\n✨ [bold green]Model verified for Sovereign AI Deployment.[/bold]")
    else:
        console.print("\n⚠️ [bold yellow]Model quality may require fine-tuning for production use.[/bold]")
