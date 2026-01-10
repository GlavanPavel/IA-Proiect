import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

from DataRead import load_data_csv_module, normalize_data, normalize_data_test, split_data
from GeneticAlgorithm import GeneticAlgorithm
from NeuralNetwork import NeuralNetwork

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class WineApp(ctk.CTk):
    def __init__(self):
        super().__init__()


        self.title("Wine Quality Neuro-Evolution")
        self.geometry("1100x650")

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

        # select train-test ratio
        self.lbl_ratio = ctk.CTkLabel(self.sidebar_frame, text="Imparitre date in antrenare/testare: ")
        self.lbl_ratio.grid(row=3, column=0, padx=20, pady=(10, 0))
        self.split_option = ctk.CTkOptionMenu(self.sidebar_frame, values=["80/20", "70/30", "60/40", "50/50"])
        self.split_option.grid(row=4, column=0, padx=20, pady=10)

        # numar generatii
        self.lbl_gen = ctk.CTkLabel(self.sidebar_frame, text="Numar Generatii:")
        self.lbl_gen.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.entry_gen = ctk.CTkEntry(self.sidebar_frame, placeholder_text="100")
        self.entry_gen.insert(0, "100")
        self.entry_gen.grid(row=6, column=0, padx=20, pady=5)

        self.lbl_mut = ctk.CTkLabel(self.sidebar_frame, text="Rata Mutatie: 0.05")
        self.lbl_mut.grid(row=7, column=0, padx=20, pady=(10, 0))

        self.slider_mut = ctk.CTkSlider(self.sidebar_frame, from_=0.01, to=0.5, number_of_steps=49,
                                        command=self.update_mutation_label)
        self.slider_mut.set(0.05)
        self.slider_mut.grid(row=8, column=0, padx=20, pady=10)

        self.btn_start = ctk.CTkButton(self.sidebar_frame, text="START Antrenare", command=self.start_training_thread)
        self.btn_start.grid(row=9, column=0, padx=20, pady=0)

        self.btn_start_testing = ctk.CTkButton(self.sidebar_frame, text="START Testare", command=self.testing_popup,state="disabled")
        self.btn_start_testing.grid(row=10, column=0, padx=20, pady=5)

        # log
        self.log_box = ctk.CTkTextbox(self.sidebar_frame, width=200, height=150)
        self.log_box.grid(row=11, column=0, padx=10, pady=10, sticky="nsew")
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

        # incarca datele
        try:
            self.log(f"Incarc fisierul: {filename}")
            raw_data = load_data_csv_module(filename)
            train_test = {
                "80/20": 0.8,
                "70/30": 0.7,
                "60/40": 0.6,
                "50/50": 0.5
            }
            data_train, self.data_test = split_data(raw_data, train_test[self.split_option.get()])
            inputs, targets, self.x_min, self.x_max = normalize_data(data_train)
        except Exception as e:
            self.log(f"Eroare fisier: {e}")
            self.btn_start.configure(state="normal")
            return

        input_size = inputs.shape[1]
        hidden_size = 15
        output_size = 1

        self.log(f"Input: {input_size}, Hidden: {hidden_size}")

        self.nn = NeuralNetwork(input_size, hidden_size, output_size)
        ga = GeneticAlgorithm(self.nn, pop_size=50, mutation_rate=mutation_rate)

        history_mse = []

        for generation in range(generations):
            current_error = ga.evolve(inputs, targets)
            history_mse.append(current_error)

            # update log fiecare 10 generatii
            if generation % 10 == 0:
                self.log(f"Gen {generation}: MSE={current_error:.4f}")

            if current_error < 0.2:
                self.log("Eroare acceptabila atinsa!")
                break

        final_scores = ga.evaluate_fitness(inputs, targets)
        best_idx = np.argmax(final_scores)
        self.best_weights = ga.population[best_idx]

        self.log(f"Finalizat. MSE Final: {history_mse[-1]:.4f}")

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
        if self.data_test is None or len(self.predictions) == 0:
            return
        if self.start_idx + 100 < len(self.data_test):
            self.start_idx += 100
            self.update_plot()
    def to_left(self):
        if self.data_test is None:
            return
        if self.start_idx - 100 >= 0:
            self.start_idx -= 100
            self.update_plot()

    def update_plot(self):

        self.ax.clear()
        self.ax.plot(self.targets, label='Rezultat asteptat', color='purple', linewidth=2)
        self.ax.plot(self.predictions, label='Rezultat obtinut', color='cyan', linewidth=2)
        self.ax.set_title(f'Rezultat testare')
        self.ax.set_xlabel('Test')
        self.ax.set_ylabel('Rezultat')
        self.ax.grid(True)
        self.ax.legend()

        end_idx = min(self.start_idx + 100, len(self.targets))
        self.ax.set_xlim(self.start_idx, end_idx)
        self.canvas.draw()

        self.btn_start.configure(state="normal")
        self.btn_start_testing.configure(state="normal")

    def testing_popup(self):
        if self.nn is None:
            self.log("Nu exista un model antrenat")
            return
        self.btn_start.configure(state="disabled")
        self.btn_start_testing.configure(state="disabled")
        self.btn_left.configure(state="normal")
        self.btn_right.configure(state="mormal")
        self.log("-" * 20)
        self.log("Incepe testarea...")

        self.inputs, self.targets = normalize_data_test(self.data_test, self.x_min, self.x_max)
        self.predictions = self.nn.forward(self.inputs, self.best_weights)

        mse_test = np.mean((self.targets - self.predictions) ** 2)
        self.log(f"Test finalizat. MSE: {mse_test:.4f}")

        self.start_idx = 0
        self.update_plot()



if __name__ == "__main__":
    app = WineApp()
    app.mainloop()