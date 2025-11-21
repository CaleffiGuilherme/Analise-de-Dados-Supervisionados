import data_handler
import model_builder
import plotter
import tensorflow as tf

EPOCHS = 100       
BATCH_SIZE = 32

def main():
    print("=== SISTEMA MULTI-MODELO DE PREDIÇÃO ACADÊMICA ===")
    
    X_train, X_test, y_train, y_test = data_handler.load_and_process_data()
    input_shape = X_train.shape[1]

    ativacoes_para_testar = ['relu', 'tanh', 'sigmoid', 'elu']
    
    resultados = {}

    for act in ativacoes_para_testar:
        print(f"\n>>> Treinando modelo com ativação: {act.upper()}...")
        
        model = model_builder.build_mlp_model(input_shape, activation_func=act)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0 
        )
        
        acc_final = history.history['val_accuracy'][-1]
        print(f"    [OK] Finalizado. Acurácia Teste: {acc_final:.4f}")
        
        resultados[act.upper()] = history

    print("\n[INFO] Gerando gráfico comparativo com todas as funções...")
    plotter.plot_results(resultados)
    
    print("\n=== FIM ===")

if __name__ == "__main__":
    main()