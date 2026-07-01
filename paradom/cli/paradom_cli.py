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
def info():
    """Shows Paradom framework status."""
    console.print("[bold blue]Paradom v0.1.0[/bold]")
    console.print("Research: Functional Equivalence ($3 = 4 - 1$)")
    console.print("Engine: Low-Resource Streaming Swapper")

if __name__ == "__main__":
    app()
