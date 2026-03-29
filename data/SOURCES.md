# Corpus — Source Documentation

Source documents indexed in the RAG knowledge base. All are official publications of the Banco Central do Brasil (BCB).

| Alias | Filename | document_id | Source URL |
|-------|----------|-------------|------------|
| Manual de Uso da Marca Pix v1.6 | `01_manual_uso_marca_pix.pdf` | `01_manual_uso_marca_pix` | [bcb.gov.br](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/marca_pix/Manual_de_Uso_da_Marca_Pix.pdf) |
| Manual de Padroes para Iniciacao do Pix v2.9.0 | `02_manual_padroes_iniciacao_pix.pdf` | `02_manual_padroes_iniciacao_pix` | [bcb.gov.br](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/Manual_Padroes_Iniciacao_Pix.pdf) |
| Manual de Fluxos de Efetivacao do Pix v2.1 | `03_manual_fluxos_efetivacao_pix.pdf` | `03_manual_fluxos_efetivacao_pix` | [bcb.gov.br](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/Manual_Fluxos_Efetivacao_Pix.pdf) |
| Requisitos Minimos para Experiencia do Usuario | `04_requisitos_minimos_experiencia_usuario.pdf` | `04_requisitos_minimos_experiencia_usuario` | [bcb.gov.br](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/Requisitos_minimos_experiencia_usuario.pdf) |
| Manual de Tempos do Pix v7.0 | `05_manual_tempos_pix.pdf` | `05_manual_tempos_pix` | [bcb.gov.br](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/IX_ManualdeTemposdoPix.pdf) |
| Manual Operacional do DICT (MED 2.0) | `06_manual_operacional_dict.pdf` | `06_manual_operacional_dict` | [bcb.gov.br](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/X_ManualOperacionaldoDICT.pdf) |
| Manual de Resolucao de Disputas v5.0 | `07_manual_resolucao_disputas.pdf` | `07_manual_resolucao_disputas` | [bcb.gov.br](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/XI_Manual_de_resolucao_de_disputa.pdf) |

---

## Not Indexed

| File | Reason |
|------|--------|
| `pix-normas` | HTML document — requires a dedicated HTML parser. Not processed by the PDF ingestion pipeline. |

---

## Re-indexing After Adding Documents

After adding new PDFs to `data/raw/`, rebuild the full pipeline:

```bash
python scripts/run_pipeline.py
```

For subsequent runs, the incremental indexer will skip unchanged chunks automatically.
