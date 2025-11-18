#!/usr/bin/env python
"""
Comprehensive verification script for TCC USP project.

Checks:
1. Date ranges for all artifacts in data_processed
2. Data sources and collection notebooks
3. Notebook audit for standardization
4. Dashboard functionality
5. Generates final report
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np

from src.io import paths as path_utils
from src.config import loader as cfg

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def check_data_file_dates(file_path: Path, date_columns: List[str]) -> Dict[str, Any]:
    """
    Check min/max dates for a data file.
    
    Args:
        file_path: Path to the data file
        date_columns: Possible date column names to check
    
    Returns:
        Dict with file info and date ranges
    """
    result = {
        "file": file_path.name,
        "exists": False,
        "min_date": None,
        "max_date": None,
        "rows": 0,
        "error": None
    }
    
    if not file_path.exists():
        return result
    
    result["exists"] = True
    
    try:
        # Determine file type and load
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
        elif file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix == ".json":
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                result["error"] = "JSON is not a list"
                return result
        else:
            result["error"] = f"Unsupported file type: {file_path.suffix}"
            return result
        
        result["rows"] = len(df)
        
        # Try to find date column
        date_col = None
        for col_name in date_columns:
            if col_name in df.columns:
                date_col = col_name
                break
        
        if date_col is None:
            # Try to infer date column
            for col in df.columns:
                if 'date' in col.lower() or 'day' in col.lower() or 'data' in col.lower():
                    date_col = col
                    break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            valid_dates = df[date_col].dropna()
            if len(valid_dates) > 0:
                result["min_date"] = valid_dates.min().strftime("%Y-%m-%d")
                result["max_date"] = valid_dates.max().strftime("%Y-%m-%d")
                result["date_column"] = date_col
        else:
            result["error"] = "No date column found"
    
    except Exception as e:
        result["error"] = str(e)
    
    return result


def verify_data_coverage() -> Dict[str, Any]:
    """Verify date coverage for all data artifacts."""
    logger.info("=" * 80)
    logger.info("ITEM 1: Verificando período efetivo dos artefatos")
    logger.info("=" * 80)
    
    data_paths = path_utils.get_data_paths(create=False)
    base_path = data_paths["base"]
    processed_path = data_paths["data_processed"]
    
    # Expected date range from config
    periodo = cfg.get_periodo_estudo()
    expected_start = periodo["start"]
    expected_end = periodo["end"]
    
    logger.info(f"Intervalo esperado (config): {expected_start} → {expected_end}")
    logger.info(f"Base path: {base_path}")
    logger.info(f"Processed path: {processed_path}")
    
    # Date column candidates
    date_cols = ["day", "date", "data", "Data", "event_day", "Date"]
    
    # Key files to check
    files_to_check = [
        "ibovespa_clean.csv",
        "news_clean.parquet",
        "news_multisource.parquet",
        "noticias_real_clean.parquet",
        "labels_y_daily.csv",
        "16_oof_predictions.csv",
        "event_study_latency.csv",
        "tfidf_daily_index.csv",
        "18_backtest_results.csv",
    ]
    
    results = []
    validation_errors = []
    
    for filename in files_to_check:
        file_path = processed_path / filename
        result = check_data_file_dates(file_path, date_cols)
        results.append(result)
        
        # Log result
        if result["exists"]:
            if result["error"]:
                logger.warning(f"  {filename}: ERRO - {result['error']}")
                validation_errors.append(f"{filename}: {result['error']}")
            elif result["min_date"] and result["max_date"]:
                in_range = (result["min_date"] >= expected_start and 
                           result["max_date"] <= expected_end)
                status = "✓" if in_range else "✗"
                logger.info(f"  {status} {filename}: {result['min_date']} → {result['max_date']} ({result['rows']} rows)")
                
                # CRITICAL: Validate date coverage matches expected range
                if not in_range:
                    msg = f"{filename}: Cobertura de datas FORA do período esperado {expected_start}→{expected_end}"
                    validation_errors.append(msg)
                
                # CRITICAL: Check for suspiciously small datasets
                if result["rows"] < 10:
                    msg = f"{filename}: Dataset muito pequeno ({result['rows']} rows) - possível dado sintético/limitado"
                    validation_errors.append(msg)
                    logger.warning(f"  ⚠ {msg}")
                    
                # CRITICAL: Check for single-day datasets (common with API fallbacks)
                if result["min_date"] == result["max_date"]:
                    msg = f"{filename}: Dataset contém apenas um dia ({result['min_date']}) - provável fallback sintético"
                    validation_errors.append(msg)
                    logger.error(f"  ✗ {msg}")
            else:
                logger.warning(f"  {filename}: Sem datas encontradas ({result['rows']} rows)")
                validation_errors.append(f"{filename}: Sem datas encontradas")
        else:
            logger.warning(f"  {filename}: ARQUIVO NÃO ENCONTRADO")
            # Don't fail for optional files
            if filename in ["ibovespa_clean.csv", "labels_y_daily.csv"]:
                validation_errors.append(f"{filename}: Arquivo crítico não encontrado")
    
    # Log summary of validation errors
    if validation_errors:
        logger.error("\n⚠️  VALIDAÇÃO FALHOU - Problemas encontrados:")
        for err in validation_errors:
            logger.error(f"    - {err}")
    else:
        logger.info("\n✅ Validação de cobertura de datas: PASSOU")
    
    return {
        "expected_range": f"{expected_start} → {expected_end}",
        "files": results,
        "validation_errors": validation_errors,
        "validation_passed": len(validation_errors) == 0
    }


def verify_data_sources() -> Dict[str, Any]:
    """Verify data sources and collection notebooks."""
    logger.info("\n" + "=" * 80)
    logger.info("ITEM 2: Verificando fontes de dados e coleta")
    logger.info("=" * 80)
    
    project_paths = path_utils.get_project_paths()
    notebooks_dir = project_paths["notebooks"]
    
    # Map notebooks to their data sources
    sources_map = {
        "00_data_download.ipynb": {
            "source": "yfinance",
            "output": "ibovespa_clean.csv",
            "description": "Download histórico Ibovespa via yfinance"
        },
        "05_data_collection_real.ipynb": {
            "source": "NewsAPI, Reuters, InfoMoney, Valor",
            "output": "noticias_real_clean.parquet",
            "description": "Coleta de notícias de fontes reais"
        },
        "12_data_collection_multisource.ipynb": {
            "source": "Multiple news sources",
            "output": "news_multisource.parquet",
            "description": "Agregação de múltiplas fontes de notícias"
        },
        "13_etl_dedup.ipynb": {
            "source": "CVM, processed news",
            "output": "news_clean.parquet",
            "description": "ETL e deduplicação de notícias"
        }
    }
    
    results = []
    for notebook_name, info in sources_map.items():
        notebook_path = notebooks_dir / notebook_name
        exists = notebook_path.exists()
        
        result = {
            "notebook": notebook_name,
            "exists": exists,
            "source": info["source"],
            "output": info["output"],
            "description": info["description"]
        }
        results.append(result)
        
        status = "✓" if exists else "✗"
        logger.info(f"  {status} {notebook_name}")
        logger.info(f"     Fonte: {info['source']}")
        logger.info(f"     Output: {info['output']}")
        logger.info(f"     Descrição: {info['description']}")
    
    return {"sources": results}


def check_notebook_imports(notebook_path: Path) -> Dict[str, Any]:
    """Check if notebook uses standardized imports."""
    result = {
        "uses_paths": False,
        "uses_config": False,
        "uses_logger": False,
        "has_obsolete_commands": False,
        "issues": []
    }
    
    if not notebook_path.exists():
        result["issues"].append("Notebook not found")
        return result
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for standardized imports
        if "from src.io import paths" in content or "import src.io.paths" in content:
            result["uses_paths"] = True
        
        if "from src.config import loader" in content or "import src.config.loader" in content:
            result["uses_config"] = True
        
        if "from src.utils import logger" in content or "import src.utils.logger" in content:
            result["uses_logger"] = True
        
        # Check for obsolete patterns
        obsolete_patterns = [
            ("C:/Users/", "Hardcoded Windows path"),
            ("/content/drive/", "Hardcoded Colab path without paths.py"),
        ]
        
        for pattern, description in obsolete_patterns:
            if pattern in content:
                result["has_obsolete_commands"] = True
                result["issues"].append(description)
    
    except Exception as e:
        result["issues"].append(f"Error reading notebook: {str(e)}")
    
    return result


def audit_notebooks() -> Dict[str, Any]:
    """Audit all notebooks 00-20."""
    logger.info("\n" + "=" * 80)
    logger.info("ITEM 3: Auditando notebooks 00-20")
    logger.info("=" * 80)
    
    project_paths = path_utils.get_project_paths()
    notebooks_dir = project_paths["notebooks"]
    
    notebook_sequence = [
        "00_data_download",
        "01_preprocessing",
        "02_baseline_logit",
        "03_tfidf_models",
        "04_embeddings_models",
        "05_data_collection_real",
        "06_preprocessing_real",
        "07_tfidf_real",
        "08_embeddings_real",
        "09_lstm_real",
        "10_dashboard_results",
        "11_event_study_latency",
        "12_data_collection_multisource",
        "13_etl_dedup",
        "14_preprocess_ptbr",
        "15_features_tfidf_daily",
        "16_models_tfidf_baselines",
        "17_sentiment_validation",
        "18_backtest_simulation",
        "19_future_extension",
        "20_final_dashboard_analysis",
    ]
    
    results = []
    for nb_name in notebook_sequence:
        nb_path = notebooks_dir / f"{nb_name}.ipynb"
        audit = check_notebook_imports(nb_path)
        
        result = {
            "notebook": nb_name,
            "exists": nb_path.exists(),
            **audit
        }
        results.append(result)
        
        # Log summary
        if not nb_path.exists():
            logger.warning(f"  ✗ {nb_name}: NÃO ENCONTRADO")
        else:
            status = "✓" if (audit["uses_paths"] and audit["uses_config"]) else "⚠"
            issues_str = f" ({len(audit['issues'])} issues)" if audit["issues"] else ""
            logger.info(f"  {status} {nb_name}{issues_str}")
            if audit["issues"]:
                for issue in audit["issues"]:
                    logger.info(f"     - {issue}")
    
    return {"notebooks": results}


def generate_report(verification_results: Dict[str, Any]) -> str:
    """Generate final verification report."""
    logger.info("\n" + "=" * 80)
    logger.info("ITEM 5: Gerando relatório final")
    logger.info("=" * 80)
    
    project_paths = path_utils.get_project_paths()
    reports_dir = project_paths["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"verification_report_{timestamp}.md"
    
    # Build report
    lines = [
        "# Relatório de Verificação Integral do Projeto TCC USP",
        f"\n**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Período esperado:** {verification_results['data_coverage']['expected_range']}",
        "\n## 1. Cobertura de Dados (data_processed)",
        "\n| Arquivo | Existe | Min Date | Max Date | Rows | Status |",
        "|---------|--------|----------|----------|------|--------|"
    ]
    
    for file_info in verification_results['data_coverage']['files']:
        exists = "✓" if file_info['exists'] else "✗"
        min_date = file_info.get('min_date', 'N/A')
        max_date = file_info.get('max_date', 'N/A')
        rows = file_info.get('rows', 0)
        error = file_info.get('error', '')
        status = error if error else "OK"
        
        lines.append(f"| {file_info['file']} | {exists} | {min_date} | {max_date} | {rows} | {status} |")
    
    lines.extend([
        "\n## 2. Fontes de Dados e Coleta",
        "\n| Notebook | Existe | Fonte | Output | Descrição |",
        "|----------|--------|-------|--------|-----------|"
    ])
    
    for source_info in verification_results['data_sources']['sources']:
        exists = "✓" if source_info['exists'] else "✗"
        lines.append(
            f"| {source_info['notebook']} | {exists} | {source_info['source']} | "
            f"{source_info['output']} | {source_info['description']} |"
        )
    
    lines.extend([
        "\n## 3. Auditoria de Notebooks",
        "\n| Notebook | Existe | paths.py | config | Issues |",
        "|----------|--------|----------|--------|--------|"
    ])
    
    for nb_info in verification_results['notebook_audit']['notebooks']:
        exists = "✓" if nb_info['exists'] else "✗"
        uses_paths = "✓" if nb_info.get('uses_paths') else "✗"
        uses_config = "✓" if nb_info.get('uses_config') else "✗"
        issues = len(nb_info.get('issues', []))
        
        lines.append(f"| {nb_info['notebook']} | {exists} | {uses_paths} | {uses_config} | {issues} |")
    
    lines.extend([
        "\n## 4. Dashboard (app_dashboard.py)",
        "\nStatus do dashboard será verificado separadamente via execução manual.",
        "\n## 5. Recomendações",
        "\n### Arquivos Ausentes"
    ])
    
    missing_files = [f['file'] for f in verification_results['data_coverage']['files'] if not f['exists']]
    if missing_files:
        for file in missing_files:
            lines.append(f"- {file}")
    else:
        lines.append("\n✓ Todos os arquivos esperados foram encontrados.")
    
    lines.append("\n### Notebooks com Issues")
    
    notebooks_with_issues = [
        nb for nb in verification_results['notebook_audit']['notebooks']
        if nb.get('issues') and nb['exists']
    ]
    
    if notebooks_with_issues:
        for nb in notebooks_with_issues:
            lines.append(f"\n**{nb['notebook']}:**")
            for issue in nb['issues']:
                lines.append(f"- {issue}")
    else:
        lines.append("\n✓ Nenhum issue crítico encontrado nos notebooks.")
    
    # Write report
    report_content = "\n".join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Relatório salvo em: {report_path}")
    
    return report_path


def main():
    """Main verification function."""
    logger.info("Iniciando verificação integral do projeto...")
    
    results = {}
    
    # 1. Verify data coverage
    results['data_coverage'] = verify_data_coverage()
    
    # 2. Verify data sources
    results['data_sources'] = verify_data_sources()
    
    # 3. Audit notebooks
    results['notebook_audit'] = audit_notebooks()
    
    # 4. Dashboard check (will be done separately)
    logger.info("\n" + "=" * 80)
    logger.info("ITEM 4: Dashboard será verificado separadamente")
    logger.info("=" * 80)
    logger.info("Execute: python app_dashboard.py")
    
    # 5. Generate report
    report_path = generate_report(results)
    
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICAÇÃO CONCLUÍDA")
    logger.info("=" * 80)
    logger.info(f"Relatório: {report_path}")
    
    return results


if __name__ == "__main__":
    main()
