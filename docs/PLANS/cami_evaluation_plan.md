## Revised evaluation plan (computational realism)

### Experimento 1 (prioridade máxima): Validação externa em CAMI II
- Cascade vs Kraken2, Kaiju, MetaPhlAn, geNomad, vConTACT3
- Datasets: CAMI II marine, strain madness
- Métricas: purity, completeness, F1 macro, UniFrac weighted, coverage
- Custo: ~1-2 semanas GPU + 2 semanas análise

### Experimento 2 (custo zero): Análise threshold-coverage
- Mesmo modelo, varia threshold de poda
- Curvas precision vs coverage por nível taxonômico
- Demonstra a "graceful degradation" da cascata
- Custo: horas de inferência adicional

### Experimento 3 (ablation seletiva): Balanceamento em sub-árvore
- Regenera Caudovirales (~150 heads) com e sem balanceamento
- Treina ambos, compara F1 macro
- Limitação reconhecida: sub-conjunto representativo
- Custo: ~3-5 dias GPU

### Experimento 4 (análise estrutural): Justificativa do Op3
- Estatísticas: o que aconteceria sem Op3
  - Cardinalidade máxima de heads sem o bucketing
  - Heads que viraríam intratáveis (>1000 classes)
- Argumento estrutural, não empírico
- Custo: algumas horas de análise

### Experimento 5 (ambicioso): Submissão pública ao portal CAMI
- Resultado citável permanentemente
- Custo: tempo de preparação dos outputs no formato CAMI

## Componentes justificados teoricamente (sem ablation):
- NoiseFilter: containers NCBI não são clados biológicos
- Cap absoluto: previne explosão fora da faixa testada do DNABERT-2
- Distribuição proporcional por folha: adaptação documentada do mestrado

Total estimated GPU time: 3-4 weeks (vs 12+ weeks for full ablation)