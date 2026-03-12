# Corpus — Source Documentation

Source documents indexed in the RAG knowledge base. All are official publications of the Banco Central do Brasil (BCB).

| Alias | Filename | Version | Source URL |
|-------|----------|---------|------------|
| Manual Operacional do DICT (MED 2.0) | `manual_operacional_dict.pdf` | MED 2.0 | [bcb.gov.br](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/X_ManualOperacionaldoDICT.pdf) |
| Manual de Tempos do Pix v7.0 | `manual_tempos_pix.pdf` | v7.0 | [bcb.gov.br](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/IX_ManualdeTemposdoPix.pdf) |
| Manual das Interfaces de Comunicação v1.12 | `manual_interfaces_comunicacao.pdf` | v1.12 | [bcb.gov.br](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/Manual_das_Interfaces_de_Comunicacao.pdf) |
| Manual de Resolução de Disputas v5.0 | `manual_resolucao_disputas.pdf` | v5.0 | [bcb.gov.br](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/XI_Manual_de_resolucao_de_disputa.pdf) |
| Manual de Fluxos do Pix v2.1 | `manual_fluxos_pix.pdf` | v2.1 | [bcb.gov.br](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/Manual_Fluxos_Efetivacao_Pix.pdf) |
| Manual de Padrões para Iniciação do Pix v2.9.0 | `manual_iniciacao_pix.pdf` | v2.9.0 | [bcb.gov.br](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/Manual_Padroes_Iniciacao_Pix.pdf) |
| Manual de Uso da Marca Pix v1.6 | `manual_marca_pix.pdf` | v1.6 | [bcb.gov.br](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/marca_pix/Manual_de_Uso_da_Marca_Pix.pdf) |

---

## Not Indexed

| File | Reason |
|------|--------|
| `pix-normas` | HTML document — requires a dedicated HTML parser. Not processed by the PDF ingestion pipeline. |

---

## Re-indexing After Adding Documents

After adding new PDFs to `data/raw/`, rebuild the full pipeline:

```bash
make pipeline   # or: python scripts/run_pipeline.py
```

This clears and rebuilds the Weaviate collection with all documents.
