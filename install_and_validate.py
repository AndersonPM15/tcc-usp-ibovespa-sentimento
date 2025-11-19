"""
Script de instalação e validação final do projeto TCC
Executa os 3 passos para atingir 100% de aderência ao plano de pesquisa
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 80)
    print(" " * 20 + "🚀 INSTALAÇÃO E VALIDAÇÃO FINAL")
    print("=" * 80)
    print()
    
    # PASSO 1: Instalar modelo spaCy
    print("📦 PASSO 1/3 - Instalando modelo spaCy pt_core_news_lg...")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", "pt_core_news_lg"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutos
        )
        
        if result.returncode == 0:
            print("✅ Modelo spaCy instalado com sucesso!")
            print(result.stdout)
        else:
            print("⚠️ Aviso durante instalação:")
            print(result.stderr)
            
            # Verificar se já está instalado
            try:
                import spacy
                nlp = spacy.load('pt_core_news_lg')
                print("✅ Modelo já estava instalado e está funcionando!")
            except:
                print("❌ Erro: Modelo não pôde ser carregado")
                return False
    except subprocess.TimeoutExpired:
        print("⚠️ Timeout na instalação (pode estar rodando em background)")
    except Exception as e:
        print(f"⚠️ Erro na instalação: {e}")
        print("Tentando validar se modelo já existe...")
        try:
            import spacy
            nlp = spacy.load('pt_core_news_lg')
            print("✅ Modelo já instalado!")
        except:
            print("❌ Modelo não disponível - instale manualmente:")
            print("   python -m spacy download pt_core_news_lg")
            return False
    
    print()
    
    # PASSO 2: Informar sobre pipeline
    print("📊 PASSO 2/3 - Pipeline de Coleta de Dados")
    print("-" * 80)
    print("⚠️ IMPORTANTE: Pipeline de coleta não será executado automaticamente")
    print("   (pode levar 30-60 minutos e requer chaves de API)")
    print()
    print("Para executar manualmente:")
    print("   python run_pipeline_multisource.py")
    print()
    print("Ou execute os notebooks individuais:")
    print("   - notebooks/12_data_collection_multisource.ipynb")
    print("   - notebooks/13_etl_dedup.ipynb")
    print()
    
    # PASSO 3: Validar instalações
    print("✅ PASSO 3/3 - Validando Instalações")
    print("-" * 80)
    
    validations = []
    
    # Validar spaCy
    try:
        import spacy
        nlp = spacy.load('pt_core_news_lg')
        print("✅ spaCy pt_core_news_lg: OK")
        validations.append(True)
    except Exception as e:
        print(f"❌ spaCy pt_core_news_lg: ERRO - {e}")
        validations.append(False)
    
    # Validar módulos criados
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    modules_to_test = [
        ("src.config.constants", "Constantes Globais"),
        ("src.utils.trading_calendar", "Calendário B3"),
        ("src.utils.validation", "TimeSeriesSplitWithEmbargo"),
        ("src.utils.preprocess_ptbr", "Lematização PT-BR"),
        ("src.utils.etl_dedup", "ETL e Deduplicação"),
    ]
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {description}: OK")
            validations.append(True)
        except Exception as e:
            print(f"❌ {description}: ERRO - {e}")
            validations.append(False)
    
    print()
    print("=" * 80)
    
    if all(validations):
        print("🎉 TODAS AS VALIDAÇÕES PASSARAM!")
        print()
        print("📋 PRÓXIMO PASSO:")
        print("   1. Execute o pipeline de coleta (se ainda não executou):")
        print("      python run_pipeline_multisource.py")
        print()
        print("   2. Execute o notebook de validação:")
        print("      jupyter notebook notebooks/21_validation_tests_research_plan.ipynb")
        print("      (Cell > Run All)")
        print()
        print("✅ Resultado esperado: 5/5 testes PASS (100%)")
        return True
    else:
        failed = sum(1 for v in validations if not v)
        print(f"⚠️ {failed}/{len(validations)} validações falharam")
        print("Revise os erros acima e corrija antes de prosseguir")
        return False
    
    print("=" * 80)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
