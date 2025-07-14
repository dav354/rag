#!/usr/bin/env python3

import os
import sys
import re
from collections import defaultdict
from urllib.parse import urlparse

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo import errors as pymongo_errors
from rich.console import Console
from rich.rule import Rule
from rich.text import Text
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

def load_config():
    """Loads configuration from environment variables or .env file."""
    load_dotenv()
    config = {
        "mongo_user": os.getenv("MONGO_USER", "scraper"),
        "mongo_pass": os.getenv("MONGO_PASS", "password"),
        "mongo_db_name": os.getenv("MONGO_DB_NAME", "askthws_scraper"),
        "mongo_host": os.getenv("MONGO_HOST", "localhost"),
        "mongo_port": int(os.getenv("MONGO_PORT", 27017)),
        "files_collection": os.getenv("MONGO_FILES_COLLECTION", "files"),
    }
    return config


def get_db_connection(config, console):
    """Establishes and returns a MongoDB database connection."""
    mongo_uri = f"mongodb://{config['mongo_user']}:{config['mongo_pass']}@{config['mongo_host']}:{config['mongo_port']}/?authSource=admin"
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command("ismaster")
        db = client[config["mongo_db_name"]]
        console.print(f"[bold green]✅ Successfully connected to MongoDB:[/bold green] {config['mongo_host']}:{config['mongo_port']}, DB: [cyan]{config['mongo_db_name']}[/cyan]")
        return db
    except pymongo_errors.ConnectionFailure as e:
        console.print(f"[bold red]Error:[/bold red] Could not connect to MongoDB at {config['mongo_host']}:{config['mongo_port']}", style="error")
        console.print(f"Details: {e}", style="error")
        sys.exit(1)
    except pymongo_errors.OperationFailure as e:
        console.print("[bold red]Error:[/bold red] MongoDB authentication failed or operation denied.", style="error")
        console.print(f"Details: {e}", style="error")
        sys.exit(1)

def analyze_pdfs_by_path(db, files_collection_name, console):
    """
    Finds all PDFs, groups them by their URL directory path, and returns the aggregated data.
    """
    console.print(f"\n[cyan]Analyzing PDFs in collection: '{files_collection_name}'...[/cyan]")
    path_data = defaultdict(lambda: {"count": 0, "filenames": []})
    
    try:
        pdf_docs_cursor = db[files_collection_name].find(
            {"type": "pdf"},
            {"url": 1}
        )
        total_pdfs = db[files_collection_name].count_documents({"type": "pdf"})

        if total_pdfs == 0:
            console.print("[yellow]No PDF documents found in the collection.[/yellow]")
            return None

        progress_columns = [
            TextColumn("[progress.description]{task.description}"), BarColumn(),
            TextColumn("[bold blue]{task.completed}/{task.total}"), TextColumn("•"),
            TimeElapsedColumn()
        ]
        with Progress(*progress_columns, console=console) as progress:
            task = progress.add_task("Processing PDFs...", total=total_pdfs)

            for doc in pdf_docs_cursor:
                url = doc.get("url")
                if not url:
                    progress.update(task, advance=1)
                    continue

                try:
                    parsed_url = urlparse(url)
                    path_key = os.path.dirname(parsed_url.path)
                    
                    if not path_key.endswith('/'):
                        path_key += '/'

                    filename = os.path.basename(parsed_url.path) or url.split('/')[-2]

                    path_data[path_key]["count"] += 1
                    path_data[path_key]["filenames"].append(filename)
                
                except Exception:
                    path_data["/ungueltige_urls/"]["count"] += 1
                    path_data["/ungueltige_urls/"]["filenames"].append(url)

                progress.update(task, advance=1)

        return path_data

    except pymongo_errors.PyMongoError as e:
        console.print(f"[bold red]Error querying database:[/bold red] {e}", style="error")
        return None


def print_report(path_data, console):
    """Prints the final, formatted report to the console as a table."""
    if not path_data:
        return

    console.print()
    
    table = Table(
        title="PDF Analysis Report by URL Path",
        show_header=True,
        header_style="bold magenta",
        show_edge=True,
        box=None,
    )
    
    table.add_column("URL Path", style="cyan", no_wrap=False, width=60)
    table.add_column("Filename", style="white")
    
    total_files = 0

    for path, data in sorted(path_data.items()):
        count = data["count"]
        filenames = sorted(data["filenames"])
        total_files += count

        table.add_section()
        table.add_row(f"[bold]{path}[/bold]", f"[bold]({count} PDF{'s' if count != 1 else ''})[/bold]")

        for filename in filenames:
            table.add_row("  →", filename)
            
    table.title = f"PDF Analysis Report by URL Path ({total_files} Total Files)"
    
    console.print(table)
    console.print(Rule("[bold green]End of Report[/bold green]"))


if __name__ == "__main__":
    console = Console()
    config = load_config()
    db = get_db_connection(config, console)
    
    pdf_data = analyze_pdfs_by_path(db, config["files_collection"], console)
    
    if pdf_data:
        print_report(pdf_data, console)