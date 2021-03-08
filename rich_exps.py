from rich.progress import track
import time
from rich.progress import Progress

from rich.progress import (
    BarColumn,
    DownloadColumn,
    TextColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
    Progress,
    TaskID,
)


progress = Progress(
    TextColumn("[bold blue]{task.fields[A]}", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    DownloadColumn(),
    "•",
    TransferSpeedColumn(),
    "•",
    TimeRemainingColumn(),
)


# progress = Progress(
#     "[progress.description]{task.description}",
#     BarColumn(),
#     "[progress.percentage]{task.percentage:>3.0f}%",
#     TimeRemainingColumn(),
# )

A = 0
with progress:

    task1 = progress.add_task("[red]Downloading...", A=A, total=1000)
    task2 = progress.add_task("[green]Processing...", A=A, total=1000)
    task3 = progress.add_task("[cyan]Cooking...", A=A, total=1000)

    while not progress.finished:
        A += 1
        progress.update(task1, advance=0.5)
        progress.update(task2, advance=0.3)
        progress.update(task3, advance=0.9)
        time.sleep(0.02)