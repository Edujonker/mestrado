# Cacheiro Viajante

Este repositório contém o script `otimizador_impressao.py`, que processa relatórios de volumetria de impressoras, calcula rotas logísticas e gera análises de desempenho.

## O que o script gera

Ao executar `python otimizador_impressao.py`, os seguintes arquivos são criados:

- `log/distance_matrix.csv`
  - Matriz de distâncias entre cidades (em km)
- `log/city_coordinates.csv`
  - Lista de cidades com latitude, longitude, volumetria e ociosidade calculada
- `log/route_nn.csv`
  - Ordem das cidades na rota inicial construída pelo algoritmo Nearest Neighbor
- `log/route_2opt.csv`
  - Ordem das cidades na rota final otimizada pelo algoritmo 2-Opt
- `log/2opt_log.csv`
  - Evolução do custo da rota a cada checkpoint do algoritmo 2-Opt
- `log/mc_metricas.csv`
  - Estatísticas descritivas do Monte Carlo (média, desvio, mínimo, percentis, máximo)
- `log/mc_summary.csv`
  - Resumo estatístico do Monte Carlo: média, mediana, desvio, min, max, assimetria, curtose, N e IC 99,9%
- `log/sa_runs.csv`
  - Resultados das 30 execuções de Simulated Annealing por seed, com custo final, melhor custo e tempo
- `log/sa_summary.csv`
  - Resumo estatístico do SA (mínimo, mediana, média, desvio, melhor custo observado)
- `log/ga_runs.csv`
  - Resultados das 30 execuções de Genetic Algorithm por seed, com custo final, melhor custo, diversidade e tempo
- `log/ga_summary.csv`
  - Resumo estatístico do AG (mínimo, mediana, média, desvio, diversidade média)
- `log/comparativo_metaheuristicas.csv`
  - Comparativo entre métodos estocásticos e roteirização, incluindo custos representativos
- `log/seeds_summary.csv`
  - Resumo das sementes usadas em cada técnica
- `log/execution_config.csv`
  - Parâmetros e sementes usados na execução
- `log/logs_consolidados.pdf`
  - PDF consolidado com todas as tabelas de log

## Observações sobre os logs

- `log/2opt_evolucao.csv` registra checkpoints do 2-Opt, incluindo iteração, custo em km, melhoria percentual relativa ao custo inicial da solução Nearest Neighbor e tempo acumulado em segundos.
- `log/sa_resumo.csv` e `log/ga_resumo.csv` registram o resultado final de cada uma das 30 execuções independentes, com seed, custo final e outras métricas específicas.
- `log/sa_summary.csv` agrupa o SA em estatísticas de mínimos, mediana, média, desvio-padrão e melhor custo observado entre as execuções.
- `log/ga_summary.csv` agrupa o AG em estatísticas de mínimos, mediana, média, desvio-padrão e diversidade final (%) medida como a proporção de indivíduos únicos na população final.
- `log/comparativo_metaheuristicas.csv` consolida custos representativos para Aleatório, 2-Opt, SA e AG e permite comparar redução percentual em relação ao custo médio do Monte Carlo.

## Interpretação dos resultados

- O algoritmo **2-Opt** apresentou o melhor equilíbrio entre qualidade e simplicidade, sendo a solução mais robusta para otimização da rota final.
- O **Algoritmo Genético** mostrou alta variabilidade entre execuções, com desvio-padrão de custo em torno de 162,47 km, mas a análise comparativa correta deve usar a mediana para representar essa distribuição.
- O teste de hipótese baseado nos resultados do **Monte Carlo** mostrou-se bem executado; o valor de t de 5.711,08 indica um efeito estatisticamente significativo.

## Como executar

1. Verifique se o arquivo de entrada `relatorios_impressoras.csv` está na pasta do projeto.
2. Execute:

```bash
python otimizador_impressao.py
```

3. Os logs ficarão dentro da pasta `log/`.

## Dependências

O script usa as seguintes bibliotecas Python:

- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `fpdf`
- `geopy`

Você pode instalar todas com:

```bash
pip install pandas numpy matplotlib scipy fpdf geopy
```

## Observações

- O diretório de logs foi configurado como `log/`.
- O PDF `log/logs_consolidados.pdf` contém todas as tabelas de log em um único documento.
- Se quiser apenas os arquivos CSV, basta abrir os arquivos dentro da pasta `log/`.
