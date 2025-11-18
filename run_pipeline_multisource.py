"""
Script auxiliar para executar o pipeline completo com validações
Facilita a execução do pipeline 12-15 (multisource) seguido de validações
"""
import sys
import subprocess
from pathlib import Path

# Adicionar src ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.validation.check_multisource import run_full_validation
from src.validation.check_pipeline_health import check_pipeline_health


def print_header(text: str):
    """Imprime cabeçalho formatado"""
    print("\n" + "="*80)
    print(text.center(80))
    print("="*80 + "\n")


def run_multisource_pipeline():
    """Executa notebooks 12-15 do pipeline multisource"""
    print_header("EXECUTANDO PIPELINE MULTISOURCE (NB 12-15)")
    
    notebooks = ["12", "13", "14", "15"]
    
    for nb in notebooks:
        print(f"\n🚀 Executando Notebook {nb}...")
        
        try:
            result = subprocess.run(
                [sys.executable, "pipeline_orchestration.py", "--only", nb],
                capture_output=True,
                text=True,
                check=True
            )
            print(result.stdout)
            print(f"✅ Notebook {nb} concluído")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erro ao executar Notebook {nb}:")
            print(e.stderr)
            return False
    
    return True


def run_validations():
    """Executa todas as validações"""
    print_header("EXECUTANDO VALIDAÇÕES")
    
    # 1. Validação multisource
    print("\n📋 1/2: Validação Multisource")
    try:
        multisource_result = run_full_validation(min_years=3, min_volume=5000)
        
        if multisource_result['status'] == 'FAILED':
            print("\n❌ Validação multisource FALHOU")
            return False
        elif multisource_result['status'] == 'WARNING':
            print("\n⚠️ Validação multisource com AVISOS")
        else:
            print("\n✅ Validação multisource PASSOU")
    
    except Exception as e:
        print(f"\n❌ Erro na validação multisource: {e}")
        return False
    
    # 2. Validação de saúde do pipeline
    print("\n📋 2/2: Saúde do Pipeline")
    try:
        health_result = check_pipeline_health()
        
        if health_result['status'] == 'UNHEALTHY':
            print("\n⚠️ Pipeline com problemas")
        elif health_result['status'] == 'WARNING':
            print("\n⚠️ Pipeline com avisos")
        else:
            print("\n✅ Pipeline saudável")
    
    except Exception as e:
        print(f"\n❌ Erro na verificação de saúde: {e}")
        return False
    
    return True


def main():
    """Função principal"""
    print_header("🚀 PIPELINE MULTISOURCE + VALIDAÇÕES")
    
    print("Este script irá:")
    print("  1. Executar notebooks 12-15 (coleta multisource → TF-IDF)")
    print("  2. Validar dados multisource (fontes, cobertura, volume)")
    print("  3. Verificar saúde geral do pipeline")
    print()
    
    # Confirmar execução
    response = input("Deseja continuar? (S/n): ").strip().lower()
    if response and response not in ['s', 'sim', 'y', 'yes']:
        print("❌ Execução cancelada pelo usuário")
        return
    
    # Executar pipeline
    if not run_multisource_pipeline():
        print("\n❌ Pipeline falhou. Abortando validações.")
        sys.exit(1)
    
    # Executar validações
    if not run_validations():
        print("\n❌ Validações falharam")
        sys.exit(1)
    
    # Sucesso
    print_header("✅ PIPELINE E VALIDAÇÕES CONCLUÍDOS COM SUCESSO")
    print("\nPróximos passos:")
    print("  - Revisar relatórios de validação em reports/")
    print("  - Executar notebooks de modelagem (16-20)")
    print("  - Verificar dashboard (notebook 10 ou 20)")


if __name__ == "__main__":
    main()
