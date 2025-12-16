# Final Runtime Checks

## py_compile
OK

## pytest -q
..                                                                       [100%]
============================== warnings summary ===============================
venv\Lib\site-packages\dash\development\base_component.py:454
  C:\TCC_USP\tcc-usp-ibovespa-sentimento\venv\Lib\site-packages\dash\development\base_component.py:454: DeprecationWarning:
  
  
  The dash_table.DataTable will be removed from the builtin dash components in a future major version.
  We recommend using dash-ag-grid as a replacement. Install with `pip install dash[ag-grid]`.

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
2 passed, 1 warning in 5.33s


## probe
[DEBUG] DEFAULT_START=2019-08-08 DEFAULT_END=2024-12-27
[DEBUG] MODEL_OPTIONS carregados: ['logreg_l2', 'rf_200']
[DEBUG] RESULTS_DF shape: (4, 7)
[DEBUG] IBOV_DF shape: (1737, 10)
[DEBUG] SENTIMENT_DF shape: (1341, 4)
PORT 127.0.0.1:8050 OPEN=True
HTTP 127.0.0.1:8050 STATUS=200
