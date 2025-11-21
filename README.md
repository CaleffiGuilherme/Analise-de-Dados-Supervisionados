# 🎓 Predição de Desempenho Acadêmico com Redes Neurais (MLP)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Este projeto implementa uma **Rede Neural Multicamadas (MLP)** para prever a aprovação ou reprovação de estudantes do ensino secundário com base em dados socioeconômicos e comportamentais.

O trabalho foi desenvolvido como parte da avaliação da disciplina de **Sistemas Inteligentes** do curso de Ciência da Computação (Universidade Positivo).

## 📊 Sobre o Projeto

O objetivo é classificar estudantes em duas categorias: **Aprovado (G3 >= 10)** ou **Reprovado (G3 < 10)**, utilizando o *Student Performance Data Set* (UCI ID 320).

O diferencial deste projeto é a arquitetura modular e a análise comparativa entre **quatro funções de ativação**:
* **ReLU** (Rectified Linear Unit)
* **Tanh** (Tangente Hiperbólica)
* **Sigmoid**
* **ELU** (Exponential Linear Unit)

## 📂 Estrutura de Arquivos

O código foi refatorado seguindo princípios de *Clean Code* e modularização:

```text
📁 Projeto_MLP/
│
├── 📄 main.py              # Script principal (Orquestrador)
├── 📄 data_handler.py      # Pipeline de ETL (Download, Limpeza e Normalização)
├── 📄 model_builder.py     # Fábrica de Redes Neurais (TensorFlow/Keras)
├── 📄 plotter.py           # Geração de gráficos comparativos
├── 📄 requirements.txt     # Lista de dependências
└── 📄 README.md            # Documentação do projeto
