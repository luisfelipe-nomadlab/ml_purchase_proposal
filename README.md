EVA Model - Análise Preditiva de Comportamento de Compra
Este projeto desenvolve um pipeline de aprendizado supervisionado para classificação binária do comportamento de compra de clientes. A proposta é avaliar modelos como Árvore de Decisão, aplicando técnicas robustas de pré-processamento, validação cruzada e análise de métricas.

Objetivo
Criar modelos capazes de prever se um cliente irá ou não adquirir um produto/serviço com base em atributos comportamentais e demográficos.

Estrutura do Projeto
eva_model.ipynb: Notebook principal contendo todo o processo de modelagem, avaliação e otimização.
costumer_processed.csv (referência): Arquivo com dados pré-processados, pronto para modelagem (utilizado via Google Drive no notebook).
Principais Etapas
Importação e Análise Exploratória dos Dados
Pré-processamento
Tratamento de dados nulos
Codificação de variáveis categóricas
Divisão dos Dados
Treinamento e teste com estratificação
Modelagem
Árvores de Decisão
Random Forest
Validação
GridSearchCV para ajuste de hiperparâmetros
Curvas ROC e análise de threshold
Avaliação
Matriz de confusão
AUC, F1-score, Precisão, Recall
Bootstrap para medir estabilidade
Tecnologias e Bibliotecas
Python 3.x
pandas, numpy
matplotlib, seaborn
scikit-learn
Google Colab (ambiente sugerido para execução)
Resultados Esperados
Modelos com desempenho balanceado entre sensibilidade e especificidade.
Otimização de threshold para maior aderência ao objetivo de negócio.
Avaliação da robustez via simulações de bootstrap.
Como Executar
Clone o repositório:
git clone https://github.com/seu_usuario/nome-do-repositorio.git
