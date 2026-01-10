import numpy as np
import matplotlib.pyplot as plt  
from DataRead import load_data_csv_module, normalize_data
from GeneticAlgorithm import GeneticAlgorithm
from NeuralNetwork import NeuralNetwork

raw_data = load_data_csv_module('winequality-red.csv')
inputs, targets, x_min, x_max = normalize_data(raw_data)

print(f"Date incarcate: {inputs.shape[0]} vinuri.")
print(f"Numar caracteristici (input_size): {inputs.shape[1]}")

input_size = 11
hidden_size = 15
output_size = 1

nn = NeuralNetwork(input_size, hidden_size, output_size)
ga = GeneticAlgorithm(nn, pop_size=50, mutation_rate=0.05)

history_mse = [] 

for generation in range(100):
    current_error= ga.evolve(inputs, targets)
    
    history_mse.append(current_error)

    if current_error < 0.2:
        print("eroare acceptabila atinsa")
        break

final_scores = ga.evaluate_fitness(inputs, targets)
best_idx = np.argmax(final_scores)
best_weights = ga.population[best_idx]

# folosind cel mai bun rezultat se fac prezicerile finale la note
final_predictions = nn.forward(inputs, best_weights)

print(f"{'Nr.':<5} | {'Real (Target)':<15} | {'Prezis (Network)':<18} | {'Diferenta':<10}")
print("-" * 55)

random_indices = np.random.choice(len(inputs), 15, replace=False)

for i in random_indices:
    real_val = targets[i][0]
    pred_val = final_predictions[i][0]
    diff = abs(real_val - pred_val)

    print(f"{i:<5} | {real_val:<15.1f} | {pred_val:<18.2f} | {diff:<10.2f}")

mean_diff = np.mean(np.abs(targets - final_predictions))
print("-" * 55)
print(f"Eroarea Medie Absoluta pe tot setul: {mean_diff:.4f}")
print(f"(In medie, reteaua greseste nota cu +/- {mean_diff:.2f} puncte)")

#graficul
plt.figure(figsize=(10, 6))
plt.plot(history_mse, label='Eroare (MSE)', color='blue')
plt.title('Evolutia erorilor pe parcusul generatiilor')
plt.xlabel('Generatii')
plt.ylabel('Valoarea MSE')
plt.grid(True)
plt.legend()
plt.show()