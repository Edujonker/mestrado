# Cacheiro Viajante

Este repositório contém dois scripts em Python para otimização de rotas logísticas de impressoras e análise de desempenho:

- `otimizador_impressao.py`
- `otimizador_pcam.py`

## Visão geral

- `otimizador_impressao.py` processa `relatorios_impressoras.csv`, gera a matriz de distâncias, cria a rota inicial com Nearest Neighbor, otimiza a rota com 2-Opt, calcula métricas de Monte Carlo, executa Simulated Annealing e Genetic Algorithm, e gera relatórios em CSV e PDF.
- `otimizador_pcam.py` roda análises paralelas de Monte Carlo, Simulated Annealing e Genetic Algorithm usando os dados gerados pelo script principal.

## Requisitos

Use o arquivo de requisitos já presente no repositório para instalar todas as dependências de uma vez:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_pcam.txt
```

Se a ativação do PowerShell estiver bloqueada, instale diretamente com:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_pcam.txt
```

O `requirements_pcam.txt` já inclui versões compatíveis, evitando conflitos comuns no ambiente global.

## Como executar

### 1. Otimizador principal

1. Coloque `relatorios_impressoras.csv` na pasta do projeto.
2. Execute:

```powershell
python otimizador_impressao.py
```

3. Verifique os arquivos gerados em `log/`.

### 2. PCAM paralelo

1. Certifique-se de que `log/distance_matrix.csv` e `log/route_nn.csv` já existem (gerados pelo `otimizador_impressao.py`).
2. Execute:

```powershell
python otimizador_pcam.py
```

3. Verifique os resultados em `log_pcam/`.

## Arquivos gerados

### Logs do otimizador principal

- `log/distance_matrix.csv`
- `log/city_coordinates.csv`
- `log/route_nn.csv`
- `log/route_2opt.csv`
- `log/2opt_log.csv`
- `log/mc_metricas.csv`
- `log/mc_summary.csv`
- `log/sa_runs.csv`
- `log/sa_summary.csv`
- `log/ga_runs.csv`
- `log/ga_summary.csv`
- `log/comparativo_metaheuristicas.csv`
- `log/seeds_summary.csv`
- `log/execution_config.csv`
- `log/logs_consolidados.pdf`

### Logs do PCAM paralelo

- `log_pcam/pcam_mc_runs.csv`
- `log_pcam/pcam_sa_runs.csv`
- `log_pcam/pcam_ga_runs.csv`

## Interpretação dos resultados

- O algoritmo **2-Opt** entrega uma solução eficiente e estável para a rota final, com checkpoints claros de evolução de custo.
- O **Genetic Algorithm** tende a ser mais variável entre execuções; por isso, a mediana é uma medida robusta para comparação.
- O **Monte Carlo** fornece uma linha de base estatística e ajuda a validar a significância das melhorias encontradas.
- O **PCAM paralelo** facilita comparar estabilidade e qualidade de Monte Carlo, SA e GA em diferentes seeds.

## Dependências

As dependências são:

- `pandas>=1.5.3,<3.0.0`
- `numpy>=2.4.6,<3.0.0`
- `matplotlib>=3.10`
- `scipy>=1.17`
- `fpdf>=1.7.2`
- `geopy>=2.4.1`

Instale todas as bibliotecas com:

```powershell
python -m pip install -r requirements_pcam.txt
```

## Observações

- Use `python -m pip install -r requirements_pcam.txt` para evitar conflitos de ambiente.
- O `log_pcam/` é criado apenas pelo `otimizador_pcam.py`.
- Se o PowerShell bloquear `Activate.ps1`, utilize o caminho completo para o Python dentro de `.venv`.
- Consulte `README_pcam.md` para mais detalhes sobre o script paralelo.
