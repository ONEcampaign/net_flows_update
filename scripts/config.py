from pathlib import Path


class Paths:
    """Class to store the paths to the data and output folders."""

    project = Path(__file__).resolve().parent.parent
    raw_data = project / "raw_data"
    output = project / "output"
    scripts = project / "scripts"
    models = scripts / "models"


CONSTANT_BASE_YEAR = 2023
ANALYSIS_YEARS: tuple = (2000, 2023)
