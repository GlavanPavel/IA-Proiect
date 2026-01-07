import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

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
        self.btn_start.grid(row=7, column=0, padx=20, pady=30)

        # log
        self.log_box = ctk.CTkTextbox(self.sidebar_frame, width=200, height=150)
        self.log_box.grid(row=8, column=0, padx=10, pady=10, sticky="nsew")
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
            inputs, targets = normalize_data(raw_data)
        except Exception as e:
            self.log(f"Eroare fisier: {e}")
            self.btn_start.configure(state="normal")
            return

        input_size = inputs.shape[1]
        hidden_size = 15
        output_size = 1

        self.log(f"Input: {input_size}, Hidden: {hidden_size}")

        nn = NeuralNetwork(input_size, hidden_size, output_size)
        ga = GeneticAlgorithm(nn, pop_size=50, mutation_rate=mutation_rate)

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


if __name__ == "__main__":
    app = WineApp()
    app.mainloop()