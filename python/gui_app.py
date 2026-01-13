import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

from DataRead import load_data_csv_module, normalize_data
from GeneticAlgorithm import GeneticAlgorithm
from NeuralNetwork import NeuralNetwork

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class WineApp(ctk.CTk):
    def __init__(self):
        super().__init__()


        self.title("Wine Quality Neuro-Evolution")
        self.geometry("1100x600")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # panou stanga
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Setări Antrenare",
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # select dataset
        self.lbl_dataset = ctk.CTkLabel(self.sidebar_frame, text="Tip Vin:")
        self.lbl_dataset.grid(row=1, column=0, padx=20, pady=(10, 0))
        self.dataset_option = ctk.CTkOptionMenu(self.sidebar_frame,
                                                values=["winequality-red.csv", "winequality-white.csv"])
        self.dataset_option.grid(row=2, column=0, padx=20, pady=10)


        # numar generatii
        self.lbl_gen = ctk.CTkLabel(self.sidebar_frame, text="Numar Generatii:")
        self.lbl_gen.grid(row=3, column=0, padx=20, pady=(10, 0))
        self.entry_gen = ctk.CTkEntry(self.sidebar_frame, placeholder_text="100")
        self.entry_gen.insert(0, "100")
        self.entry_gen.grid(row=4, column=0, padx=20, pady=5)

        self.lbl_mut = ctk.CTkLabel(self.sidebar_frame, text="Rata Mutatie: 0.05")
        self.lbl_mut.grid(row=5, column=0, padx=20, pady=(10, 0))

        self.slider_mut = ctk.CTkSlider(self.sidebar_frame, from_=0.01, to=0.5, number_of_steps=49,
                                        command=self.update_mutation_label)
        self.slider_mut.set(0.05)
        self.slider_mut.grid(row=6, column=0, padx=20, pady=10)

        self.btn_start = ctk.CTkButton(self.sidebar_frame, text="START Antrenare", command=self.start_training_thread)
        self.btn_start.grid(row=7, column=0, padx=20, pady=10)

        self.btn_start_testing = ctk.CTkButton(self.sidebar_frame, text="Vizualizare rezultate", command=self.view_results,state="disabled")
        self.btn_start_testing.grid(row=8, column=0, padx=20, pady=10)

        # log
        self.log_box = ctk.CTkTextbox(self.sidebar_frame, width=200, height=160)
        self.log_box.grid(row=9, column=0, padx=10, pady=20, sticky="nsew")
        self.log_box.insert("0.0", "Așteptare comandă...\n")

        # panou dreapta
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # placeholder pentru grafic
        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Evolutia Erorii (MSE)")
        self.ax.set_xlabel("Generatii")
        self.ax.set_ylabel("Eroare")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.move_frame = ctk.CTkFrame(self.right_frame)
        self.move_frame.pack(side="top", pady=10)

        self.btn_left = ctk.CTkButton(self.move_frame, text="<<", command=self.to_left, state="disabled")
        self.btn_left.pack(side="left", padx=10)
        self.btn_right = ctk.CTkButton(self.move_frame, text=">>", command=self.to_right, state="disabled")
        self.btn_right.pack(side="left", padx=10)


    def update_mutation_label(self, value):
        self.lbl_mut.configure(text=f"Rată Mutație: {value:.2f}")

    def log(self, message):
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")

    def start_training_thread(self):
        thread = threading.Thread(target=self.run_training)
        thread.start()

    def run_training(self):
        self.btn_start.configure(state="disabled")
        self.btn_start_testing.configure(state="disabled")
        self.btn_left.configure(state="disabled")
        self.btn_right.configure(state="disabled")
        self.nn = None
        self.log("-" * 20)
        self.log("Incepe antrenarea...")

        filename = self.dataset_option.get()
        try:
            generations = int(self.entry_gen.get())
            mutation_rate = self.slider_mut.get()
        except ValueError:
            self.log("Eroare")
            self.btn_start.configure(state="normal")
            return

        if generations <= 0:
            self.log("Eroare: Numarul de generatii trebuie sa fie mai mare de 0")
            self.btn_start.configure(state="normal")
            return

        # incarca datele
        try:
            self.log(f"Incarc fisierul: {filename}")
            raw_data = load_data_csv_module(filename)
            self.inputs, self.targets = normalize_data(raw_data)
        except Exception as e:
            self.log(f"Eroare fisier: {e}")
            self.btn_start.configure(state="normal")
            return

        input_size = self.inputs.shape[1]
        hidden_size = 15
        output_size = 1

        self.log(f"Input: {input_size}, Hidden: {hidden_size}")

        start_time = time.time()
        self.nn = NeuralNetwork(input_size, hidden_size, output_size)
        ga = GeneticAlgorithm(self.nn, pop_size=50, mutation_rate=mutation_rate)

        history_mse = []

        for generation in range(generations):
            current_error = ga.evolve(self.inputs, self.targets)
            history_mse.append(current_error)

            # update log fiecare 10 generatii
            if generation % 10 == 0:
                self.log(f"Gen {generation}: MSE={current_error:.4f}")

            if current_error < 0.2:
                self.log("Eroare acceptabila atinsa!")
                break

        scores = ga.evaluate_fitness(self.inputs, self.targets)
        best_idx = np.argmax(scores)
        self.best_weights = ga.population[best_idx]

        exc_time = time.time() - start_time

        self.log(f"Finalizat. MSE Final: {history_mse[-1]:.4f}")
        self.log(f"Timp executie: {exc_time:.2f} secunde")
        self.ax.clear()
        self.ax.plot(history_mse, label='Eroare (MSE)', color='cyan')
        self.ax.set_title(f'Rezultat: {filename} (Mutatie: {mutation_rate:.2f})')
        self.ax.set_xlabel('Generații')
        self.ax.set_ylabel('MSE')
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

        self.btn_start.configure(state="normal")
        self.btn_start_testing.configure(state="normal")

    def to_right(self):
        if self.nn is None or len(self.predictions) == 0:
            return
        if self.start_idx + 100 < len(self.predictions):
            self.start_idx += 100
            self.update_plot()
    def to_left(self):
        if self.nn is None:
            return
        if self.start_idx - 100 >= 0:
            self.start_idx -= 100
            self.update_plot()

    def update_plot(self):

        self.ax.clear()
        self.ax.plot(self.targets, label='Rezultat asteptat', color='purple', linewidth=2)
        self.ax.plot(self.predictions, label='Rezultat obtinut', color='cyan', linewidth=2)
        self.ax.set_title(f'Rezultate antrenare')
        self.ax.set_xlabel('Index')
        self.ax.set_ylabel('Scor')
        self.ax.grid(True)
        self.ax.legend()

        end_idx = min(self.start_idx + 100, len(self.targets))
        self.ax.set_xlim(self.start_idx, end_idx)
        self.canvas.draw()


    def view_results(self):
        if self.nn is None:
            self.log("Nu exista un model antrenat")
            return
        self.btn_start.configure(state="disabled")
        self.btn_start_testing.configure(state="disabled")
        self.btn_left.configure(state="normal")
        self.btn_right.configure(state="normal")
        self.log("-" * 20)

        self.predictions = np.round(self.nn.forward(self.inputs, self.best_weights))
        corect = np.sum(self.targets - self.predictions == 0)
        accuracy = corect / len(self.targets)*100
        self.log(f"Rezultatul este atins in {accuracy:.2f}% din cazuri")
        self.start_idx = 0
        self.update_plot()

        self.btn_start.configure(state="normal")
        self.btn_start_testing.configure(state="normal")



if __name__ == "__main__":
    app = WineApp()
    app.mainloop()