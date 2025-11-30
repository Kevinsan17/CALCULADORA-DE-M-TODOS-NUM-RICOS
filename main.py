import customtkinter as ctk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import calculos as calc

# Configuración visual global
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

ANCHO_APP = 850
ALTO_NORMAL = 450
ALTO_GRAFICA = 650

estado_graficas = {"Regresión": False, "Regresión Polinomial": False}

# ======================================================
# SISTEMAS DE ECUACIONES LINEALES
# ======================================================


def configurar_pestana_gauss_simple(padre):
    ctk.CTkLabel(padre, text="Gauss Simple", font=(
        "Arial", 20, "bold")).pack(pady=10)
    ctk.CTkLabel(padre, text="Matriz A (filas ; cols ,)").pack()
    entry_A = ctk.CTkEntry(padre, width=300)
    entry_A.pack(pady=5)
    ctk.CTkLabel(padre, text="Vector b (comas)").pack()
    entry_b = ctk.CTkEntry(padre, width=300)
    entry_b.pack(pady=5)
    lbl_res = ctk.CTkLabel(padre, text="", font=(
        "Consolas", 14), text_color="#4CC9F0")
    lbl_res.pack(pady=15)

    def calcular():
        try:
            lbl_res.configure(text="")
            x = calc.gauss_simple(entry_A.get(), entry_b.get())
            lbl_res.configure(text=f"Solución:\n{x}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    ctk.CTkButton(padre, text="Calcular", command=calcular).pack()


def configurar_pestana_gauss(padre):
    ctk.CTkLabel(padre, text="Gauss-Seidel",
                 font=("Arial", 20, "bold")).pack(pady=10)
    ctk.CTkLabel(padre, text="Matriz A (filas ; cols ,)").pack()
    entry_A = ctk.CTkEntry(padre, width=300, placeholder_text="Ej: 3,1; 1,4")
    entry_A.pack(pady=5)
    ctk.CTkLabel(padre, text="Vector b (comas)").pack()
    entry_b = ctk.CTkEntry(padre, width=300, placeholder_text="Ej: 5,6")
    entry_b.pack(pady=5)
    ctk.CTkLabel(padre, text="x0 (comas)").pack()
    entry_x0 = ctk.CTkEntry(padre, width=300, placeholder_text="Ej: 0,0")
    entry_x0.pack(pady=5)

    lbl_warning = ctk.CTkLabel(padre, text="", font=(
        "Arial", 12), text_color="orange")
    lbl_warning.pack(pady=5)

    lbl_resultado = ctk.CTkLabel(padre, text="", font=(
        "Consolas", 14), text_color="#4CC9F0")
    lbl_resultado.pack(pady=10)

    def calcular():
        try:
            lbl_resultado.configure(text="")
            lbl_warning.configure(text="")

            es_segura, filas_malas = calc.verificar_diagonal_dominante(
                entry_A.get())

            if not es_segura:
                lbl_warning.configure(
                    text=f"⚠️ CUIDADO: La matriz NO es diagonal dominante.\nFilas problemáticas: {filas_malas}\nEs probable que el método no converja."
                )
            x = calc.gauss_seidel(entry_A.get(), entry_b.get(), entry_x0.get())
            lbl_resultado.configure(text=f"Solución:\n{x}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    ctk.CTkButton(padre, text="Calcular", command=calcular).pack()


def configurar_pestana_pivoteo(padre):
    ctk.CTkLabel(padre, text="Pivoteo Parcial",
                 font=("Arial", 20, "bold")).pack(pady=10)
    ctk.CTkLabel(padre, text="Matriz A (filas ; cols ,)").pack()
    entry_A = ctk.CTkEntry(padre, width=300)
    entry_A.pack(pady=5)
    ctk.CTkLabel(padre, text="Vector b (comas)").pack()
    entry_b = ctk.CTkEntry(padre, width=300)
    entry_b.pack(pady=5)
    lbl_resultado = ctk.CTkLabel(padre, text="", font=(
        "Consolas", 14), text_color="#4CC9F0")
    lbl_resultado.pack(pady=15)

    def calcular():
        try:
            lbl_resultado.configure(text="")
            x = calc.pivoteo_parcial(entry_A.get(), entry_b.get())
            lbl_resultado.configure(text=f"Solución:\n{x}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    ctk.CTkButton(padre, text="Calcular", command=calcular).pack()

# ======================================================
# ECUACIONES NO LINEALES
# ======================================================


def configurar_pestana_biseccion(padre):
    ctk.CTkLabel(padre, text="Método de Bisección",
                 font=("Arial", 20, "bold")).pack(pady=10)
    ctk.CTkLabel(padre, text="Función f(x):").pack()
    entry_fx = ctk.CTkEntry(padre, width=300)
    entry_fx.pack(pady=5)
    frame_datos = ctk.CTkFrame(padre, fg_color="transparent")
    frame_datos.pack(pady=5)
    ctk.CTkLabel(frame_datos, text="a:").grid(row=0, column=0, padx=5)
    entry_a = ctk.CTkEntry(frame_datos, width=80)
    entry_a.grid(row=1, column=0, padx=5)
    ctk.CTkLabel(frame_datos, text="b:").grid(row=0, column=1, padx=5)
    entry_b = ctk.CTkEntry(frame_datos, width=80)
    entry_b.grid(row=1, column=1, padx=5)
    ctk.CTkLabel(frame_datos, text="Tol:").grid(row=0, column=2, padx=5)
    entry_tol = ctk.CTkEntry(frame_datos, width=80)
    entry_tol.grid(row=1, column=2, padx=5)
    lbl_res = ctk.CTkLabel(padre, text="", font=(
        "Consolas", 14), text_color="#4CC9F0")
    lbl_res.pack(pady=15)

    def calcular():
        try:
            lbl_res.configure(text="")
            xr, iteraciones = calc.biseccion(entry_fx.get(), float(
                entry_a.get()), float(entry_b.get()), float(entry_tol.get()))
            lbl_res.configure(text=f"Raíz: {xr:.6f} (Iter: {iteraciones})")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    ctk.CTkButton(padre, text="Calcular", command=calcular).pack()


def configurar_pestana_newton(padre):
    ctk.CTkLabel(padre, text="Newton-Raphson",
                 font=("Arial", 20, "bold")).pack(pady=10)
    ctk.CTkLabel(padre, text="Función f(x):").pack()
    entry_fx = ctk.CTkEntry(padre, width=300)
    entry_fx.pack(pady=5)
    ctk.CTkLabel(padre, text="Derivada f'(x):").pack()
    entry_dfx = ctk.CTkEntry(padre, width=300)
    entry_dfx.pack(pady=5)
    frame_datos = ctk.CTkFrame(padre, fg_color="transparent")
    frame_datos.pack(pady=5)
    ctk.CTkLabel(frame_datos, text="x0:").grid(row=0, column=0, padx=5)
    entry_x0 = ctk.CTkEntry(frame_datos, width=100)
    entry_x0.grid(row=1, column=0, padx=5)
    ctk.CTkLabel(frame_datos, text="Tol:").grid(row=0, column=1, padx=5)
    entry_tol = ctk.CTkEntry(frame_datos, width=100)
    entry_tol.grid(row=1, column=1, padx=5)
    lbl_res = ctk.CTkLabel(padre, text="", font=(
        "Consolas", 14), text_color="#4CC9F0")
    lbl_res.pack(pady=15)

    def calcular():
        try:
            lbl_res.configure(text="")
            xr, iteraciones = calc.newton_raphson(
                entry_fx.get(), entry_dfx.get(), float(entry_x0.get()), float(entry_tol.get()))
            lbl_res.configure(text=f"Raíz: {xr:.6f} (Iter: {iteraciones})")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    ctk.CTkButton(padre, text="Calcular", command=calcular).pack()


def configurar_pestana_secante(padre):
    ctk.CTkLabel(padre, text="Método de la Secante",
                 font=("Arial", 20, "bold")).pack(pady=10)
    ctk.CTkLabel(padre, text="Función f(x):").pack()
    entry_fx = ctk.CTkEntry(padre, width=300)
    entry_fx.pack(pady=5)
    frame_datos = ctk.CTkFrame(padre, fg_color="transparent")
    frame_datos.pack(pady=5)
    ctk.CTkLabel(frame_datos, text="x0:").grid(row=0, column=0, padx=5)
    entry_a = ctk.CTkEntry(frame_datos, width=80)
    entry_a.grid(row=1, column=0, padx=5)
    ctk.CTkLabel(frame_datos, text="x1:").grid(row=0, column=1, padx=5)
    entry_b = ctk.CTkEntry(frame_datos, width=80)
    entry_b.grid(row=1, column=1, padx=5)
    ctk.CTkLabel(frame_datos, text="Tol:").grid(row=0, column=2, padx=5)
    entry_tol = ctk.CTkEntry(frame_datos, width=80)
    entry_tol.grid(row=1, column=2, padx=5)
    lbl_resultado = ctk.CTkLabel(padre, text="", font=(
        "Consolas", 14), text_color="#4CC9F0")
    lbl_resultado.pack(pady=15)

    def calcular():
        try:
            lbl_resultado.configure(text="")
            xr = calc.secante(entry_fx.get(), float(entry_a.get()), float(
                entry_b.get()), float(entry_tol.get()))
            lbl_resultado.configure(text=f"Raíz: {xr:.6f}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    ctk.CTkButton(padre, text="Calcular", command=calcular).pack()


def configurar_pestana_secante_mod(padre):
    ctk.CTkLabel(padre, text="Secante Modificada",
                 font=("Arial", 20, "bold")).pack(pady=10)
    ctk.CTkLabel(padre, text="Función f(x):").pack()
    entry_fx = ctk.CTkEntry(padre, width=300)
    entry_fx.pack(pady=5)
    frame_datos = ctk.CTkFrame(padre, fg_color="transparent")
    frame_datos.pack(pady=5)
    ctk.CTkLabel(frame_datos, text="x0:").grid(row=0, column=0, padx=5)
    entry_x0 = ctk.CTkEntry(frame_datos, width=80)
    entry_x0.grid(row=1, column=0, padx=5)
    ctk.CTkLabel(frame_datos, text="delta:").grid(row=0, column=1, padx=5)
    entry_d = ctk.CTkEntry(frame_datos, width=80)
    entry_d.grid(row=1, column=1, padx=5)
    ctk.CTkLabel(frame_datos, text="Tol:").grid(row=0, column=2, padx=5)
    entry_tol = ctk.CTkEntry(frame_datos, width=80)
    entry_tol.grid(row=1, column=2, padx=5)
    lbl_res = ctk.CTkLabel(padre, text="", font=(
        "Consolas", 14), text_color="#4CC9F0")
    lbl_res.pack(pady=15)

    def calcular():
        try:
            lbl_res.configure(text="")
            xr, iter = calc.secante_modificada(entry_fx.get(), float(
                entry_x0.get()), float(entry_d.get()), float(entry_tol.get()))
            lbl_res.configure(text=f"Raíz: {xr:.6f} (Iter: {iter})")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    ctk.CTkButton(padre, text="Calcular", command=calcular).pack()

# ======================================================
# REGRESIONES (LÓGICA GRÁFICA + CÁLCULO EXTERNO)
# ======================================================


def configurar_pestana_regresion(padre):
    ctk.CTkLabel(padre, text="Regresión Lineal",
                 font=("Arial", 20, "bold")).pack(pady=10)
    frame_entradas = ctk.CTkFrame(padre, fg_color="transparent")
    frame_entradas.pack(pady=5)
    ctk.CTkLabel(frame_entradas, text="Datos X:").grid(row=0, column=0, padx=5)
    entry_x = ctk.CTkEntry(frame_entradas, width=250)
    entry_x.grid(row=0, column=1, padx=5)
    ctk.CTkLabel(frame_entradas, text="Datos Y:").grid(
        row=1, column=0, padx=5, pady=5)
    entry_y = ctk.CTkEntry(frame_entradas, width=250)
    entry_y.grid(row=1, column=1, padx=5, pady=5)
    lbl_resultado = ctk.CTkLabel(padre, text="", font=(
        "Consolas", 12), text_color="#4CC9F0")
    lbl_resultado.pack(pady=5)

    ctk.CTkButton(padre, text="Calcular y Graficar",
                  command=lambda: calcular()).pack(pady=10)
    frame_grafica = ctk.CTkFrame(padre, fg_color="transparent")
    frame_grafica.pack(pady=10, padx=10, fill="both", expand=True)

    def calcular():
        try:
            lbl_resultado.configure(text="")
            for widget in frame_grafica.winfo_children():
                widget.destroy()

            # Obtener resultados limpios desde calculos.py
            x, y, y_pred, ecuacion, r2 = calc.regresion_lineal(
                entry_x.get(), entry_y.get())

            lbl_resultado.configure(text=f"{ecuacion} | R² = {r2:.4f}")

            # Graficar (Esto se queda en la interfaz porque usa TkAgg)
            fig = Figure(figsize=(5, 3.5), dpi=100)
            ax = fig.add_subplot(111)
            ax.scatter(x, y, color='red', label='Datos')
            ax.plot(x, y_pred, color='blue', label='Ajuste')
            ax.set_title("Ajuste Lineal")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

            canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            estado_graficas["Regresión"] = True
            root.geometry(f"{ANCHO_APP}x{ALTO_GRAFICA}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


def configurar_pestana_regresion_poli(padre):
    ctk.CTkLabel(padre, text="Regresión Polinomial",
                 font=("Arial", 20, "bold")).pack(pady=10)
    frame_entradas = ctk.CTkFrame(padre, fg_color="transparent")
    frame_entradas.pack(pady=5)
    ctk.CTkLabel(frame_entradas, text="Datos X:").grid(row=0, column=0, padx=5)
    entry_x = ctk.CTkEntry(frame_entradas, width=200)
    entry_x.grid(row=0, column=1, padx=5)
    ctk.CTkLabel(frame_entradas, text="Datos Y:").grid(row=1, column=0, padx=5)
    entry_y = ctk.CTkEntry(frame_entradas, width=200)
    entry_y.grid(row=1, column=1, padx=5)
    ctk.CTkLabel(frame_entradas, text="Grado (n):").grid(
        row=2, column=0, padx=5, pady=5)
    entry_n = ctk.CTkEntry(frame_entradas, width=200)
    entry_n.grid(row=2, column=1, padx=5, pady=5)
    lbl_res = ctk.CTkLabel(padre, text="", font=(
        "Consolas", 12), text_color="#4CC9F0")
    lbl_res.pack(pady=5)

    ctk.CTkButton(padre, text="Calcular y Graficar",
                  command=lambda: calcular()).pack(pady=10)
    frame_grafica = ctk.CTkFrame(padre, fg_color="transparent")
    frame_grafica.pack(pady=10, padx=10, fill="both", expand=True)

    def calcular():
        try:
            lbl_res.configure(text="")
            for widget in frame_grafica.winfo_children():
                widget.destroy()

            # Obtener datos procesados desde calculos.py
            grado_in = int(entry_n.get())
            x, y, x_suave, y_suave, ec_texto, r2, grado = calc.regresion_polinomial(
                entry_x.get(), entry_y.get(), grado_in)

            lbl_res.configure(text=f"{ec_texto}\nR² = {r2:.4f}")

            # Graficar
            fig = Figure(figsize=(5, 3.5), dpi=100)
            ax = fig.add_subplot(111)
            ax.scatter(x, y, color='red', label='Datos')
            ax.plot(x_suave, y_suave, color='blue',
                    label=f'Grado {grado}')  # Usamos curva suave
            ax.set_title("Ajuste Polinomial")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

            canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            estado_graficas["Regresión Polinomial"] = True
            root.geometry(f"{ANCHO_APP}x{ALTO_GRAFICA}")
        except Exception as e:
            messagebox.showerror("Error", str(e))


# ======================================================
# CONFIGURACIÓN PRINCIPAL
# ======================================================
root = ctk.CTk()
root.title("Calculadora de Métodos Numéricos")
root.geometry(f"{ANCHO_APP}x{ALTO_NORMAL}")


def al_cambiar_pestana():
    nombre_actual = vista_pestanas.get()
    tiene_grafica = estado_graficas.get(nombre_actual, False)
    if tiene_grafica:
        root.geometry(f"{ANCHO_APP}x{ALTO_GRAFICA}")
    else:
        root.geometry(f"{ANCHO_APP}x{ALTO_NORMAL}")


vista_pestanas = ctk.CTkTabview(root, command=al_cambiar_pestana)
vista_pestanas.pack(pady=10, padx=10, fill="both", expand=True)

# Agregar pestañas
tab_gauss_simple = vista_pestanas.add("Gauss Simple")
tab_gauss = vista_pestanas.add("Gauss-Seidel")
tab_pivoteo = vista_pestanas.add("Pivoteo")
tab_biseccion = vista_pestanas.add("Bisección")
tab_newton = vista_pestanas.add("Newton-Raphson")
tab_secante = vista_pestanas.add("Secante")
tab_secante_mod = vista_pestanas.add("Secante Modificada")
tab_regresion = vista_pestanas.add("Regresión")
tab_regresion_poli = vista_pestanas.add("Regresión Polinomial")

# Configurar pestañas
configurar_pestana_gauss_simple(tab_gauss_simple)
configurar_pestana_gauss(tab_gauss)
configurar_pestana_pivoteo(tab_pivoteo)
configurar_pestana_biseccion(tab_biseccion)
configurar_pestana_newton(tab_newton)
configurar_pestana_secante(tab_secante)
configurar_pestana_secante_mod(tab_secante_mod)
configurar_pestana_regresion(tab_regresion)
configurar_pestana_regresion_poli(tab_regresion_poli)

root.mainloop()
