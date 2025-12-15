# RUNBOOK — Release Pack Dashboard

## Pré-requisitos
- Python do venv: .\\venv\\Scripts\\python.exe
- Dados locais em C:\\TCC_USP\\data_processed (período 2018-01-02 a 2024-12-31).

## Passos recomendados
1) (Opcional) Regenerar artefatos mínimos
   .\\venv\\Scripts\\python.exe scripts\\pipeline_minimal.py
2) Validação de integridade
   .\\venv\\Scripts\\python.exe scripts\\data_integrity_report.py
3) Testes
   .\\venv\\Scripts\\python.exe -m pytest -q
4) Probe do dashboard (HTTP 200 esperado)
   .\\venv\\Scripts\\python.exe app_dashboard.py --probe --host 127.0.0.1 --port 8050
5) Subir dashboard
   .\\venv\\Scripts\\python.exe app_dashboard.py --host 127.0.0.1 --port 8050 --find-port --open
6) Exportar figuras (8 gráficos)
   .\\venv\\Scripts\\python.exe scripts\\export_dashboard_figures.py

## Notas
- Nenhum dado é versionado; use sempre C:\\TCC_USP\\data_processed.
- Latência: se o dataset estiver vazio, o card é tratado academicamente (mensagem textual).
