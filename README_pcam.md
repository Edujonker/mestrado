# PCAM Parallel Execution

Este README descreve o novo script `otimizador_pcam.py`, que utiliza paralelismo para rodar componentes da metodologia PCAM sobre os dados de roteirização.

## Objetivo

O script `otimizador_pcam.py` tem como objetivo:

- executar simulações de Monte Carlo em paralelo
- executar múltiplas sementes de Simulated Annealing em paralelo
- executar múltiplas sementes de Genetic Algorithm em paralelo
- consolidar resultados em arquivos CSV no diretório `log_pcam/`

## Arquivos gerados

- `log_pcam/pcam_mc_runs.csv`
  - resultados do Monte Carlo por worker
- `log_pcam/pcam_sa_runs.csv`
  - resultados das execuções de Simulated Annealing em paralelo
- `log_pcam/pcam_ga_runs.csv`
  - resultados das execuções de Genetic Algorithm em paralelo

## Requisitos

O script usa as mesmas dependências do `otimizador_impressao.py`, com versões compatíveis para evitar conflitos de pacote:

- `pandas>=1.5.3,<3.0.0`
- `numpy>=2.4.6,<3.0.0`
- `matplotlib>=3.10`
- `scipy>=1.17`
- `fpdf>=1.7.2`
- `geopy>=2.4.1`

Recomendamos instalar em um ambiente virtual separado para não impactar outras bibliotecas instaladas no mesmo Python.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_pcam.txt
```

Se você já tiver outros pacotes instalados no ambiente global, não use `pip install pandas numpy matplotlib scipy fpdf geopy` diretamente, pois isso pode quebrar dependências de projetos existentes.

## Como executar

1. Certifique-se de que `log/distance_matrix.csv` já existe, gerado pelo `otimizador_impressao.py`.
2. Execute:

```bash
python otimizador_pcam.py
```

3. Os resultados serão escritos na pasta `log_pcam/`.

## Observações

- O paralelismo é feito com a biblioteca `multiprocessing` do Python.
- O script utiliza a função `nearest_neighbor_route` e os algoritmos de `otimizador_impressao.py`.
- Os parâmetros podem ser ajustados dentro da variável `PCAM_CONFIG` no início do arquivo.
