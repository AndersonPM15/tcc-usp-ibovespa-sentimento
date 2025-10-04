#!/bin/bash
# ==========================================================
#  Script automático de atualização do TCC - USP
#  Autor: Anderson P. M.
#  Descrição: adiciona, comita e envia atualizações para o GitHub
# ==========================================================

# Exibir cabeçalho
echo "🚀 Iniciando atualização automática do repositório TCC_USP..."
echo "==========================================================="

# Verifica se há alterações
git status

# Adiciona todos os arquivos modificados e novos
echo "📂 Adicionando todos os arquivos..."
git add .

# Cria mensagem automática de commit com data e hora
commit_message="Atualização automática em $(date '+%d/%m/%Y às %H:%M:%S')"
echo "📝 Criando commit: '$commit_message'"
git commit -m "$commit_message"

# Envia para o GitHub
echo "🌐 Enviando alterações para o GitHub..."
git push origin main || git push origin master

# Confirmação final
echo "✅ Atualização concluída com sucesso!"
echo "==========================================================="
