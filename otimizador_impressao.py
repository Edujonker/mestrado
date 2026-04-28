import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random
import time
from scipy.spatial.distance import cdist
from fpdf import FPDF
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- CONFIGURAÇÕES DE ENGENHARIA ---
CAPACIDADE_NOMINAL = 15000  # Páginas/mês por máquina
CUSTO_KM_OPERACIONAL = 3.85 # R$ por KM (Combustível + Hora Técnica)

def master_otimizador_pericia():
    arquivo = 'relatorios_impressoras.csv'
    if not os.path.exists(arquivo):
        print(f"Erro: Arquivo {arquivo} não encontrado.")
        return

    # 1. TRATAMENTO DE DADOS (Lógica de Limpeza de Relatórios)
    print("Passo 1: Limpeza e Consolidação de Volumetria...")
    df_raw = pd.read_csv(arquivo, sep=';', encoding='latin-1', header=None)
    def clean(v):
        try: return float(str(v).replace('.', '').replace(',', '.'))
        except: return 0

    dados_brutos = []
    for i in range(3, len(df_raw)):
        row = df_raw.iloc[i]
        if pd.notna(row[1]):
            # Soma volumetria total de todas as colunas de meses
            vol_total = sum([clean(row[j]) for j in range(9, 35)])
            dados_brutos.append({
                'cidade': str(row[1]).strip().upper(),
                'setor': str(row[3]).strip(),
                'vol': vol_total / 13 # Média mensal aproximada
            })
    
    df_cidades = pd.DataFrame(dados_brutos).groupby('cidade').agg({'vol': 'sum', 'setor': 'count'}).reset_index()

    # 2. GEOCODIFICAÇÃO (Mapeamento SC com Foco em Florianópolis)
    print("Passo 2: Mapeamento Geográfico (Geocoding Santa Catarina)...")
    geolocator = Nominatim(user_agent="pericia_master_sc")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    df_cidades['loc'] = df_cidades['cidade'].apply(lambda c: geocode(f"{c}, Santa Catarina, Brazil"))
    df_cidades['lat'] = df_cidades['loc'].apply(lambda l: l.latitude if l else None)
    df_cidades['lon'] = df_cidades['loc'].apply(lambda l: l.longitude if l else None)
    df_cidades = df_cidades.dropna(subset=['lat']).reset_index(drop=True)

    # Garantir Florianópolis como HUB (Origem 0)
    idx_f = df_cidades[df_cidades['cidade'].str.contains("FLORIANO")].index.tolist()
    if idx_f:
        df_cidades = pd.concat([df_cidades.iloc[idx_f], df_cidades.drop(idx_f)]).reset_index(drop=True)

    # 3. COMPETIÇÃO DE METAHEURÍSTICAS (TSP)
    coords = df_cidades[['lat', 'lon']].values
    n = len(df_cidades)
    
    # A. Vizinho Mais Próximo (Heurística de Construção)
    v_visitados = [0]; v_rota = [0]; v_dist = 0; atual = 0
    for _ in range(n - 1):
        dists = cdist([coords[atual]], coords)[0]
        dists[v_visitados] = np.inf
        prox = np.argmin(dists)
        v_dist += dists[prox]
        v_visitados.append(prox); v_rota.append(prox); atual = prox
    dist_vp_km = v_dist * 111.12

    # B. 2-Opt (Refinamento da Rota)
    rota_2opt = v_rota[:]
    melhor_d = v_dist
    for i in range(1, n - 2):
        for j in range(i + 1, n):
            nova_rota = rota_2opt[:i] + rota_2opt[i:j][::-1] + rota_2opt[j:]
            nova_d = sum(cdist([coords[nova_rota[k]]], [coords[nova_rota[k+1]]])[0][0] for k in range(n-1))
            if nova_d < melhor_d:
                melhor_d = nova_d
                rota_2opt = nova_rota
    dist_2opt_km = melhor_d * 111.12

    # C. Simulação de Monte Carlo (50.000 rotas aleatórias para prova de ROI)
    print("Passo 3: Rodando Teste de Estresse (Monte Carlo 50k)...")
    amostras = [sum(cdist([coords[r[k]]], [coords[r[k+1]]])[0][0] for k in range(n-1)) * 111.12 
                for r in [random.sample(range(n), n) for _ in range(50000)]]
    dist_media_sim = np.mean(amostras)

    # 4. GERAÇÃO DE GRÁFICOS CIENTÍFICOS
    print("Passo 4: Gerando Evidências Visuais...")
    # Gráfico 1: Mapa com Nomes e Rota
    plt.figure(figsize=(12, 8))
    plt.plot(df_cidades.iloc[rota_2opt]['lon'], df_cidades.iloc[rota_2opt]['lat'], 'g-o', linewidth=1.5)
    for i, r in df_cidades.iterrows():
        plt.text(r['lon'], r['lat'], r['cidade'], fontsize=7, fontweight='bold')
    plt.title("Malha Logística de Perícia SC (Otimização 2-Opt)")
    plt.savefig("mapa_master.png", dpi=300); plt.close()

    # Gráfico 2: Comparativo de ROI Logístico
    plt.figure(figsize=(10, 5))
    plt.bar(['Intuição (Média)', 'Vizinho Próximo', 'Otimizado (2-Opt)'], 
            [dist_media_sim, dist_vp_km, dist_2opt_km], color=['gray', 'blue', 'green'])
    plt.ylabel("Distância Total (KM)")
    plt.title("Comparativo de Eficiência Logística")
    plt.savefig("comparativo_master.png"); plt.close()

    # 5. CONSOLIDAÇÃO DO PDF (RELATÓRIO DE ARTIGO)
    print("Passo 5: Consolidando Relatório Final em PDF...")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 15, "AUDITORIA MASTER: LOGISTICA E ROI DE IMPRESSAO", ln=True, align='C')
    
    # Justificativas Acadêmicas
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. COMPLEXIDADE E JUSTIFICATIVA MATEMATICA", ln=True)
    pdf.set_font("Arial", '', 10)
    txt = (f"Para as {n} cidades mapeadas, o numero total de possibilidades de rota e fatorial "
           f"((n-1)!): {math.factorial(n-1):.2e}. A simulacao de Monte Carlo com 50.000 iteracoes "
           f"provou que a rota intuitiva media percorreria {dist_media_sim:.2f} KM, enquanto a "
           f"rota otimizada via 2-Opt percorre apenas {dist_2opt_km:.2f} KM, gerando uma "
           f"economia direta de {((dist_media_sim - dist_2opt_km)/dist_media_sim)*100:.2f}% em deslocamento.")
    pdf.multi_cell(0, 7, txt)

    pdf.add_page()
    pdf.image("mapa_master.png", x=10, y=20, w=190)
    pdf.image("comparativo_master.png", x=10, y=160, w=180)

    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. INVENTARIO, VOLUMETRIA E ANALISE DE OCIO POR CIDADE", ln=True)
    pdf.set_font("Courier", '', 7)
    pdf.cell(45, 7, "CIDADE", 1); pdf.cell(35, 7, "VOL. MEDIO", 1); pdf.cell(20, 7, "ATIVOS", 1); pdf.cell(25, 7, "% OCIO", 1); pdf.ln()
    for i in rota_2opt:
        row = df_cidades.iloc[i]
        ocio = (1 - (row['vol'] / (row['setor'] * CAPACIDADE_NOMINAL))) * 100
        pdf.cell(45, 6, row['cidade'][:20], 1)
        pdf.cell(35, 6, f"{row['vol']:,.0f}", 1)
        pdf.cell(20, 6, str(row['setor']), 1)
        pdf.cell(25, 6, f"{max(0, ocio):.1f}%", 1); pdf.ln()

    pdf.output("Artigo_Master_Finalizado.pdf")
    print("\nSUCESSO: Relatório 'Artigo_Master_Finalizado.pdf' gerado!")

if __name__ == "__main__":
    master_otimizador_pericia()