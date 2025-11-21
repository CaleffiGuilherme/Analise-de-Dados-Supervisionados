import matplotlib.pyplot as plt

def plot_results(results_dict):
    print("[INFO] Gerando gráficos comparativos múltiplos...")
    plt.figure(figsize=(16, 6))

    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    plt.subplot(1, 2, 1)
    for i, (name, history) in enumerate(results_dict.items()):
        cor = colors[i % len(colors)]
        plt.plot(history.history['loss'], label=f'{name} (Treino)', color=cor, linewidth=1.5)
        plt.plot(history.history['val_loss'], linestyle='--', color=cor, linewidth=1.0, alpha=0.7)
        
    plt.title('Comparação de Perda (Loss)')
    plt.xlabel('Épocas')
    plt.ylabel('Erro')
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for i, (name, history) in enumerate(results_dict.items()):
        cor = colors[i % len(colors)]
        plt.plot(history.history['accuracy'], label=f'{name} (Treino)', color=cor, linewidth=1.5)
        plt.plot(history.history['val_accuracy'], linestyle='--', label=f'{name} (Validação)', color=cor, linewidth=1.0)

    plt.title('Comparação de Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()