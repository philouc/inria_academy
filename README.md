# inria_academy
Inria Academy Training on advanced signal processing methods

Set of Python scripts that span basics about Fourier, (continuous/discrete) wavelet analysis and then
introduce tools for the analysis first of non-stationary signals and second, of multi-component mixed ones, 
notably the STFT and VWD for time-frequency or time-scale analysis and EMD, SST, VMD, and the multivariate VMD extension for the extraction of Implicit Mode Functions (IMF). 


## Prerequisite

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) to be awave of the virtuval environment needed

## Installation

```bash
git clone https://github.com/philouc/inria_academy.git
cd inria_academy
uv sync
```

L'environnement virtuel est créé automatiquement dans `.venv/` et toutes les
dépendances sont installées à partir de `pyproject.toml` / `uv.lock`.

## Utilisation

```bash
uv run python -m         # paradigme Go/NoGo standard
```


## Structure

```

## Documentation

THe documentation is generated using MkDocs and automatically published on
[GitHub Pages](https://philouc.github.io/inria_academy/) after every push on `main`.

Pour la prévisualiser localement :

```bash
uv run mkdocs serve
```


## Licence

[CC BY-NC-ND 4.0](LICENSE).
