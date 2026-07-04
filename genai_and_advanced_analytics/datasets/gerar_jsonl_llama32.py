#!/usr/bin/env python3
"""
Conversor de Perguntas/Respostas para JSONL formato Llama 3.2 Messages

Este script converte um CSV com perguntas e respostas em um arquivo JSONL
compatível com o formato Llama 3.2 Messages para treinamento de modelos de IA.

Uso:
    python gerar_jsonl_llama32.py --input perguntas.csv --output saida.jsonl
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any

# System prompt padrão para SynapseAI
DEFAULT_SYSTEM_PROMPT = """Você é um assistente virtual especializado da SynapseAI Solutions. A SynapseAI é uma empresa que desenvolve produtos corporativos baseados em Inteligência Artificial Generativa para os setores de Finanças, Educação e Saúde.

Seus produtos incluem:
- FinBrain: Copiloto financeiro corporativo
- RiskGen: Plataforma de risco, compliance e auditoria
- EduMentor AI: Tutor educacional inteligente
- CourseGen Studio: Geração de conteúdo educacional
- ClinicaGPT: Assistente clínico de apoio à decisão

Responda de forma clara, precisa e profissional sobre a empresa, seus produtos, tecnologias e serviços."""


def ler_csv(caminho_csv: str) -> List[Dict[str, str]]:
    """
    Lê um arquivo CSV com colunas 'pergunta' e 'resposta'.
    
    Args:
        caminho_csv: Caminho para o arquivo CSV
        
    Returns:
        Lista de dicionários com as colunas do CSV
    """
    dados = []
    try:
        with open(caminho_csv, 'r', encoding='utf-8') as f:
            leitor = csv.DictReader(f)
            for linha in leitor:
                dados.append(linha)
        print(f"✓ Lidos {len(dados)} pares de pergunta/resposta do CSV")
        return dados
    except FileNotFoundError:
        print(f"✗ Erro: Arquivo '{caminho_csv}' não encontrado")
        exit(1)
    except Exception as e:
        print(f"✗ Erro ao ler CSV: {e}")
        exit(1)


def converter_para_jsonl(
    dados: List[Dict[str, str]], 
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> List[Dict[str, Any]]:
    """
    Converte lista de perguntas/respostas para formato Llama 3.2 Messages.
    
    Args:
        dados: Lista com dicionários contendo 'pergunta' e 'resposta'
        system_prompt: Prompt do sistema a ser usado em cada exemplo
        
    Returns:
        Lista de exemplos no formato JSONL Llama 3.2 Messages
    """
    exemplos = []
    
    for idx, item in enumerate(dados, 1):
        pergunta = item.get('pergunta', '').strip()
        resposta = item.get('resposta', '').strip()
        
        # Validar
        if not pergunta or not resposta:
            print(f"⚠ Aviso: Exemplo {idx} ignorado (pergunta ou resposta vazia)")
            continue
        
        # Remover aspas extras se existirem
        pergunta = pergunta.strip('"\'')
        resposta = resposta.strip('"\'')
        
        # Criar estrutura Llama 3.2 Messages
        exemplo = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": pergunta
                },
                {
                    "role": "assistant",
                    "content": resposta
                }
            ]
        }
        
        exemplos.append(exemplo)
    
    print(f"✓ Convertidos {len(exemplos)} exemplos para formato Llama 3.2 Messages")
    return exemplos


def salvar_jsonl(exemplos: List[Dict[str, Any]], caminho_saida: str) -> None:
    """
    Salva exemplos em arquivo JSONL.
    
    Args:
        exemplos: Lista de exemplos no formato Llama 3.2 Messages
        caminho_saida: Caminho para salvar o arquivo JSONL
    """
    try:
        with open(caminho_saida, 'w', encoding='utf-8') as f:
            for exemplo in exemplos:
                # Escrever como JSON na linha
                json.dump(exemplo, f, ensure_ascii=False)
                f.write('\n')
        
        tamanho_kb = Path(caminho_saida).stat().st_size / 1024
        print(f"✓ Arquivo JSONL salvo: {caminho_saida}")
        print(f"  - {len(exemplos)} exemplos")
        print(f"  - {tamanho_kb:.1f} KB")
        
    except Exception as e:
        print(f"✗ Erro ao salvar JSONL: {e}")
        exit(1)


def validar_jsonl(caminho_jsonl: str) -> bool:
    """
    Valida se o arquivo JSONL está bem formado.
    
    Args:
        caminho_jsonl: Caminho do arquivo JSONL a validar
        
    Returns:
        True se válido, False caso contrário
    """
    try:
        linhas_validas = 0
        linhas_invalidas = 0
        
        with open(caminho_jsonl, 'r', encoding='utf-8') as f:
            for num_linha, linha in enumerate(f, 1):
                linha = linha.strip()
                if not linha:
                    continue
                
                try:
                    json.loads(linha)
                    linhas_validas += 1
                except json.JSONDecodeError:
                    print(f"✗ Linha {num_linha} contém JSON inválido")
                    linhas_invalidas += 1
        
        if linhas_invalidas == 0:
            print(f"✓ Arquivo JSONL válido ({linhas_validas} linhas)")
            return True
        else:
            print(f"✗ Arquivo JSONL inválido ({linhas_invalidas} erros)")
            return False
            
    except Exception as e:
        print(f"✗ Erro ao validar JSONL: {e}")
        return False


def main():
    """Função principal com argumentos de linha de comando."""
    
    parser = argparse.ArgumentParser(
        description='Converte CSV com perguntas/respostas em JSONL Llama 3.2 Messages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Converter arquivo CSV para JSONL
  python gerar_jsonl_llama32.py --input perguntas.csv --output saida.jsonl

  # Validar arquivo JSONL
  python gerar_jsonl_llama32.py --validate saida.jsonl

  # Usar prompt do sistema customizado
  python gerar_jsonl_llama32.py --input perguntas.csv --output saida.jsonl --system-prompt "Seu prompt aqui"
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Arquivo CSV de entrada com colunas: pergunta, resposta'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Arquivo JSONL de saída'
    )
    
    parser.add_argument(
        '--validate', '-v',
        type=str,
        help='Validar arquivo JSONL existente'
    )
    
    parser.add_argument(
        '--system-prompt', '-sp',
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help='Prompt do sistema customizado (default: prompt SynapseAI)'
    )
    
    args = parser.parse_args()
    
    # Modo validação
    if args.validate:
        print(f"\n📋 Validando arquivo JSONL: {args.validate}")
        validar_jsonl(args.validate)
        return
    
    # Modo conversão
    if not args.input or not args.output:
        print("✗ Erro: --input e --output são obrigatórios")
        print("Use -h para ajuda")
        exit(1)
    
    print(f"\n📋 Convertendo CSV para JSONL")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    
    # Processar
    dados = ler_csv(args.input)
    exemplos = converter_para_jsonl(dados, args.system_prompt)
    salvar_jsonl(exemplos, args.output)
    
    # Validar
    print(f"\n✓ Validando arquivo gerado...")
    validar_jsonl(args.output)
    
    print("\n✅ Processo concluído com sucesso!")


if __name__ == "__main__":
    main()
