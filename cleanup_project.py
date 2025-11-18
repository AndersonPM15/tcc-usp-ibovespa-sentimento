"""
Script de limpeza final do projeto
Remove outputs dos notebooks, limpa cache e prepara projeto para commit
"""
import os
import json
import shutil
from pathlib import Path
from typing import List


def clean_notebook_outputs(notebook_path: Path) -> bool:
    """
    Remove outputs de um notebook Jupyter
    
    Args:
        notebook_path: Path para o notebook
    
    Returns:
        True se limpeza foi bem-sucedida
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Limpar outputs de cada célula
        cells_cleaned = 0
        for cell in nb.get('cells', []):
            if cell.get('cell_type') == 'code':
                if cell.get('outputs'):
                    cell['outputs'] = []
                    cells_cleaned += 1
                if cell.get('execution_count'):
                    cell['execution_count'] = None
        
        # Salvar notebook limpo
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        
        if cells_cleaned > 0:
            print(f"   ✅ {notebook_path.name}: {cells_cleaned} células limpas")
        else:
            print(f"   ℹ️ {notebook_path.name}: já limpo")
        
        return True
    
    except Exception as e:
        print(f"   ❌ {notebook_path.name}: Erro ao limpar - {e}")
        return False


def clean_all_notebooks(notebooks_dir: Path) -> int:
    """
    Limpa outputs de todos os notebooks
    
    Args:
        notebooks_dir: Diretório de notebooks
    
    Returns:
        Número de notebooks limpos
    """
    print("\n📓 Limpando outputs dos notebooks...")
    print("-" * 60)
    
    notebooks = list(notebooks_dir.glob("*.ipynb"))
    notebooks = [nb for nb in notebooks if not nb.name.startswith('_')]
    
    cleaned = 0
    for nb_path in sorted(notebooks):
        if clean_notebook_outputs(nb_path):
            cleaned += 1
    
    print(f"\n✅ {cleaned}/{len(notebooks)} notebooks processados")
    return cleaned


def clean_pycache(project_root: Path) -> int:
    """
    Remove diretórios __pycache__
    
    Args:
        project_root: Raiz do projeto
    
    Returns:
        Número de diretórios removidos
    """
    print("\n🗑️ Removendo __pycache__...")
    print("-" * 60)
    
    removed = 0
    for pycache_dir in project_root.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            print(f"   ✅ Removido: {pycache_dir.relative_to(project_root)}")
            removed += 1
        except Exception as e:
            print(f"   ❌ Erro ao remover {pycache_dir}: {e}")
    
    if removed == 0:
        print("   ℹ️ Nenhum __pycache__ encontrado")
    
    print(f"\n✅ {removed} diretórios __pycache__ removidos")
    return removed


def clean_temp_files(project_root: Path) -> int:
    """
    Remove arquivos temporários (.pyc, .DS_Store, etc)
    
    Args:
        project_root: Raiz do projeto
    
    Returns:
        Número de arquivos removidos
    """
    print("\n🧹 Removendo arquivos temporários...")
    print("-" * 60)
    
    patterns = ["*.pyc", "*.pyo", ".DS_Store", "Thumbs.db", "desktop.ini"]
    removed = 0
    
    for pattern in patterns:
        for temp_file in project_root.rglob(pattern):
            try:
                temp_file.unlink()
                print(f"   ✅ Removido: {temp_file.relative_to(project_root)}")
                removed += 1
            except Exception as e:
                print(f"   ❌ Erro ao remover {temp_file}: {e}")
    
    if removed == 0:
        print("   ℹ️ Nenhum arquivo temporário encontrado")
    
    print(f"\n✅ {removed} arquivos temporários removidos")
    return removed


def verify_critical_files(project_root: Path) -> bool:
    """
    Verifica se arquivos críticos existem
    
    Args:
        project_root: Raiz do projeto
    
    Returns:
        True se todos os arquivos críticos existem
    """
    print("\n🔍 Verificando arquivos críticos...")
    print("-" * 60)
    
    critical_files = [
        "README.md",
        "requirements.txt",
        "pipeline_orchestration.py",
        "run_pipeline_multisource.py",
        "configs/config_tcc.yaml",
        "src/io/paths.py",
        "src/config/loader.py",
    ]
    
    all_exist = True
    for file_path in critical_files:
        full_path = project_root / file_path
        exists = full_path.exists()
        
        status = "✅" if exists else "❌"
        print(f"   {status} {file_path}")
        
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✅ Todos os arquivos críticos existem")
    else:
        print("\n⚠️ Alguns arquivos críticos estão faltando")
    
    return all_exist


def count_project_stats(project_root: Path) -> dict:
    """
    Conta estatísticas do projeto
    
    Args:
        project_root: Raiz do projeto
    
    Returns:
        Dict com estatísticas
    """
    print("\n📊 Estatísticas do Projeto...")
    print("-" * 60)
    
    stats = {
        'notebooks': 0,
        'python_files': 0,
        'total_lines': 0,
        'python_lines': 0,
    }
    
    # Contar notebooks
    notebooks_dir = project_root / "notebooks"
    if notebooks_dir.exists():
        stats['notebooks'] = len([nb for nb in notebooks_dir.glob("*.ipynb") 
                                  if not nb.name.startswith('_')])
    
    # Contar arquivos Python e linhas
    src_dir = project_root / "src"
    if src_dir.exists():
        for py_file in src_dir.rglob("*.py"):
            if '__pycache__' not in str(py_file):
                stats['python_files'] += 1
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # Contar apenas linhas não vazias e não comentários
                        code_lines = [l for l in lines 
                                     if l.strip() and not l.strip().startswith('#')]
                        stats['python_lines'] += len(code_lines)
                        stats['total_lines'] += len(lines)
                except:
                    pass
    
    print(f"   📓 Notebooks: {stats['notebooks']}")
    print(f"   🐍 Arquivos Python (src/): {stats['python_files']}")
    print(f"   📝 Linhas Python (total): {stats['total_lines']:,}")
    print(f"   💻 Linhas de código (sem comentários/vazias): {stats['python_lines']:,}")
    
    return stats


def generate_cleanup_report(project_root: Path, results: dict):
    """
    Gera relatório de limpeza
    
    Args:
        project_root: Raiz do projeto
        results: Dict com resultados da limpeza
    """
    print("\n" + "="*80)
    print("📄 RELATÓRIO DE LIMPEZA FINAL")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = {
        'timestamp': timestamp,
        'notebooks_cleaned': results['notebooks_cleaned'],
        'pycache_removed': results['pycache_removed'],
        'temp_files_removed': results['temp_files_removed'],
        'critical_files_ok': results['critical_files_ok'],
        'project_stats': results['project_stats']
    }
    
    # Salvar relatório
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = reports_dir / f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Relatório salvo em: {report_file}")
    
    # Resumo
    print("\n" + "="*80)
    print("✅ LIMPEZA CONCLUÍDA")
    print("="*80)
    print(f"\n✨ Projeto limpo e pronto para commit!")
    print(f"\n📊 Resumo:")
    print(f"   - {results['notebooks_cleaned']} notebooks limpos")
    print(f"   - {results['pycache_removed']} diretórios __pycache__ removidos")
    print(f"   - {results['temp_files_removed']} arquivos temporários removidos")
    print(f"   - {results['project_stats']['python_lines']:,} linhas de código Python")


def main():
    """Função principal"""
    print("="*80)
    print("🧹 LIMPEZA FINAL DO PROJETO TCC USP")
    print("="*80)
    
    # Determinar raiz do projeto
    project_root = Path(__file__).parent
    print(f"\n📁 Projeto: {project_root}")
    
    # Confirmar
    print("\n⚠️ Esta operação irá:")
    print("   1. Limpar outputs de todos os notebooks")
    print("   2. Remover diretórios __pycache__")
    print("   3. Remover arquivos temporários (.pyc, .DS_Store, etc)")
    print()
    
    response = input("Deseja continuar? (S/n): ").strip().lower()
    if response and response not in ['s', 'sim', 'y', 'yes']:
        print("❌ Limpeza cancelada pelo usuário")
        return
    
    # Executar limpeza
    results = {}
    
    # 1. Limpar notebooks
    notebooks_dir = project_root / "notebooks"
    if notebooks_dir.exists():
        results['notebooks_cleaned'] = clean_all_notebooks(notebooks_dir)
    else:
        print("\n⚠️ Diretório notebooks/ não encontrado")
        results['notebooks_cleaned'] = 0
    
    # 2. Remover __pycache__
    results['pycache_removed'] = clean_pycache(project_root)
    
    # 3. Remover arquivos temporários
    results['temp_files_removed'] = clean_temp_files(project_root)
    
    # 4. Verificar arquivos críticos
    results['critical_files_ok'] = verify_critical_files(project_root)
    
    # 5. Estatísticas do projeto
    results['project_stats'] = count_project_stats(project_root)
    
    # 6. Gerar relatório
    generate_cleanup_report(project_root, results)


if __name__ == "__main__":
    main()
