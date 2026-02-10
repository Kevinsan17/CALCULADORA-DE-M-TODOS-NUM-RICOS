import customtkinter as ctk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import calculos as calc
import numpy as np

# Configuración visual global
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ANCHO_APP = 850
ALTO_NORMAL = 650
ALTO_GRAFICA = 900

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


def al_cambiar_pestana(tabview):
    nombre_actual = tabview.get()
    tiene_grafica = estado_graficas.get(nombre_actual, False)
    if tiene_grafica:
        root.geometry(f"{ANCHO_APP}x{ALTO_GRAFICA}")
    else:
        root.geometry(f"{ANCHO_APP}x{ALTO_NORMAL}")


vista_principal = ctk.CTkTabview(root)
vista_principal.pack(pady=10, padx=10, fill="both", expand=True)

tab_parcial1 = vista_principal.add("Primer Parcial")
tab_parcial2 = vista_principal.add("Segundo Parcial")

vista_p1 = ctk.CTkTabview(tab_parcial1, command=lambda: al_cambiar_pestana(vista_p1))
vista_p1.pack(fill="both", expand=True)

vista_p2 = ctk.CTkTabview(tab_parcial2, command=lambda: al_cambiar_pestana(vista_p2))
vista_p2.pack(fill="both", expand=True)


def configurar_pestana_interpolacion(padre):
    ctk.CTkLabel(padre, text="Interpolación Polinomial",
                 font=("Arial", 20, "bold")).pack(pady=10)

    # Selector de Método
    frame_selector = ctk.CTkFrame(padre, fg_color="transparent")
    frame_selector.pack(pady=5)
    ctk.CTkLabel(frame_selector, text="Método:").grid(row=0, column=0, padx=5)
    combo_metodo = ctk.CTkOptionMenu(frame_selector, values=["Lagrange", "Newton"],
                                     command=lambda m: cambiar_interfaz(m))
    combo_metodo.grid(row=0, column=1, padx=5)

    # Frame contenedor dinámico
    frame_dinamico = ctk.CTkFrame(padre, fg_color="transparent")
    frame_dinamico.pack(fill="both", expand=True, padx=10, pady=5)

    def cambiar_interfaz(metodo):
        for widget in frame_dinamico.winfo_children():
            widget.destroy()

        if metodo == "Lagrange":
            configurar_ui_lagrange(frame_dinamico)
        else:
            configurar_ui_newton(frame_dinamico)

    # ==========================
    # UI LAGRANGE
    # ==========================
    def configurar_ui_lagrange(frame):
        # Frame Entradas
        frame_entradas = ctk.CTkFrame(frame, fg_color="transparent")
        frame_entradas.pack(pady=5)

        # Datos Principales
        ctk.CTkLabel(frame_entradas, text="Puntos X (comas):").grid(row=0, column=0, padx=5, sticky="e")
        entry_x = ctk.CTkEntry(frame_entradas, width=300, placeholder_text="Ej: 0, 1, 2, 3")
        entry_x.grid(row=0, column=1, padx=5, pady=2)

        ctk.CTkLabel(frame_entradas, text="Puntos Y (comas):").grid(row=1, column=0, padx=5, sticky="e")
        entry_y = ctk.CTkEntry(frame_entradas, width=300, placeholder_text="Ej: 1, 2, 0, 4")
        entry_y.grid(row=1, column=1, padx=5, pady=2)

        # Separador visual
        ctk.CTkLabel(frame_entradas, text="--- Validación (Opcional) ---", text_color="gray").grid(row=2, column=0, columnspan=2, pady=5)

        # Datos Validación
        ctk.CTkLabel(frame_entradas, text="Validación X (comas):").grid(row=3, column=0, padx=5, sticky="e")
        entry_val_x = ctk.CTkEntry(frame_entradas, width=300, placeholder_text="Puntos para calcular error (opcional)")
        entry_val_x.grid(row=3, column=1, padx=5, pady=2)

        ctk.CTkLabel(frame_entradas, text="Validación Y (comas):").grid(row=4, column=0, padx=5, sticky="e")
        entry_val_y = ctk.CTkEntry(frame_entradas, width=300, placeholder_text="Valores reales para comparar error")
        entry_val_y.grid(row=4, column=1, padx=5, pady=2)

        # Punto a evaluar
        ctk.CTkLabel(frame_entradas, text="Evaluar en x =").grid(row=5, column=0, padx=5, sticky="e", pady=(10,0))
        entry_x_interp = ctk.CTkEntry(frame_entradas, width=150, placeholder_text="Ej: 1.5")
        entry_x_interp.grid(row=5, column=1, padx=5, pady=(10,0), sticky="w")

        # Botón
        ctk.CTkButton(frame, text="Construir Polinomio y Calcular",
                      command=lambda: calcular_lagrange(entry_x, entry_y, entry_val_x, entry_val_y, entry_x_interp, frame_res)).pack(pady=10)

        # Área de Resultados (Scrollable para informe)
        frame_res = ctk.CTkScrollableFrame(frame, height=500, label_text="Informe de Resultados")
        frame_res.pack(fill="both", expand=True)

    def calcular_lagrange(ex, ey, evx, evy, ex_int, frame_res):
        try:
            # Limpiar resultados previos
            for widget in frame_res.winfo_children():
                widget.destroy()

            # Calcular
            res = calc.interpolacion_lagrange(ex.get(), ey.get(), float(ex_int.get()), evx.get(), evy.get())
            x, y, x_eval, y_eval, poli_texto, y_interp, pasos_str, error_str, val_data = res

            # Mostrar Polinomio y Resultado
            ctk.CTkLabel(frame_res, text="Polinomio Resultante:", font=("Arial", 14, "bold"), anchor="w").pack(fill="x", pady=(5,0))
            ctk.CTkLabel(frame_res, text=poli_texto, font=("Consolas", 12), justify="left", wraplength=750).pack(fill="x", pady=5)

            # Mostrar Pasos
            ctk.CTkLabel(frame_res, text="Construcción de Bases (Pasos):", font=("Arial", 14, "bold"), anchor="w").pack(fill="x", pady=(10,0))
            ctk.CTkLabel(frame_res, text=pasos_str, font=("Consolas", 11), justify="left", wraplength=750, text_color="gray80").pack(fill="x", pady=5)

            # Mostrar Error
            ctk.CTkLabel(frame_res, text="Análisis de Error:", font=("Arial", 14, "bold"), anchor="w").pack(fill="x", pady=(10,0))
            ctk.CTkLabel(frame_res, text=error_str, font=("Consolas", 12), justify="left", text_color="#FFA500").pack(fill="x", pady=5)

            # Graficar
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.scatter(x, y, color='red', s=80, label='Puntos datos', zorder=3)
            ax.plot(x_eval, y_eval, color='blue', linewidth=2, label='Polinomio Lagrange')
            
            # Punto evaluado
            ax.scatter([float(ex_int.get())], [y_interp], color='green', s=100, marker='*', label=f'Evaluación ({float(ex_int.get())}, {y_interp:.2f})', zorder=4)

            # Puntos de validación si existen
            if val_data:
                vx, vy = val_data
                ax.scatter(vx, vy, color='orange', marker='x', s=60, label='Puntos Validación')

            ax.set_title("Interpolación de Lagrange")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

            canvas = FigureCanvasTkAgg(fig, master=frame_res)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)

            estado_graficas["Lagrange"] = True
            root.geometry(f"{ANCHO_APP}x{ALTO_GRAFICA}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ==========================
    # UI NEWTON
    # ==========================
    def configurar_ui_newton(frame):
        frame_entradas = ctk.CTkFrame(frame, fg_color="transparent")
        frame_entradas.pack(pady=5)

        ctk.CTkLabel(frame_entradas, text="Modo:").grid(row=0, column=0, padx=5, sticky="e")
        combo_modo = ctk.CTkOptionMenu(frame_entradas, values=["General (Todos los puntos)", "Lineal (2 puntos)", "Cuadrática (3 puntos)"])
        combo_modo.grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(frame_entradas, text="Puntos X (comas):").grid(row=1, column=0, padx=5, sticky="e")
        entry_x = ctk.CTkEntry(frame_entradas, width=300)
        entry_x.grid(row=1, column=1, padx=5, pady=2)

        ctk.CTkLabel(frame_entradas, text="Puntos Y (comas):").grid(row=2, column=0, padx=5, sticky="e")
        entry_y = ctk.CTkEntry(frame_entradas, width=300)
        entry_y.grid(row=2, column=1, padx=5, pady=2)

        ctk.CTkLabel(frame_entradas, text="Evaluar en x =").grid(row=3, column=0, padx=5, sticky="e")
        entry_val = ctk.CTkEntry(frame_entradas, width=150)
        entry_val.grid(row=3, column=1, padx=5, pady=2, sticky="w")

        ctk.CTkButton(frame, text="Calcular",
                      command=lambda: calcular_newton(entry_x, entry_y, entry_val, combo_modo, frame_grafica)).pack(pady=10)

        ctk.CTkLabel(frame, text="Resultados:", font=("Arial", 14, "bold")).pack()
        lbl_res = ctk.CTkLabel(frame, text="", font=("Consolas", 12))
        lbl_res.pack(pady=5)

        frame_grafica = ctk.CTkFrame(frame, fg_color="transparent")
        frame_grafica.pack(fill="both", expand=True)

        # Referencia para actualizar el label dentro de calcular
        # Hack: guardamos referencia en el frame
        frame_grafica.lbl_res = lbl_res

    def calcular_newton(ex, ey, ev, combo, frame_graf):
        try:
            for widget in frame_graf.winfo_children(): widget.destroy()
            frame_graf.lbl_res.configure(text="")
            
            modo = combo.get()
            x_str = ex.get()
            y_str = ey.get()
            val = float(ev.get())

            # Filtrado segun modo
            x_list = x_str.split(',')
            y_list = y_str.split(',')
            
            if modo == "Lineal (2 puntos)":
                if len(x_list) < 2: raise ValueError("Requiere al menos 2 puntos")
                x_str = ",".join(x_list[:2])
                y_str = ",".join(y_list[:2])
            elif modo == "Cuadrática (3 puntos)":
                if len(x_list) < 3: raise ValueError("Requiere al menos 3 puntos")
                x_str = ",".join(x_list[:3])
                y_str = ",".join(y_list[:3])

            res = calc.interpolacion_newton(x_str, y_str, val)
            x, y, x_eval, y_eval, poli_texto, y_interp = res # Unpack correcto

            frame_graf.lbl_res.configure(text=poli_texto)

            # Graficar
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.scatter(x, y, color='red', label='Puntos')
            ax.plot(x_eval, y_eval, color='purple', label='Newton')
            ax.scatter([val], [y_interp], color='green', marker='*', s=100, label='Interpolado')
            ax.legend()
            ax.grid(True, linestyle='--')
            
            canvas = FigureCanvasTkAgg(fig, master=frame_graf)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            estado_graficas["Newton Interpolación"] = True
            root.geometry(f"{ANCHO_APP}x{ALTO_GRAFICA}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # Iniciar con Lagrange por defecto
    cambiar_interfaz("Lagrange")


def configurar_pestana_integracion(padre):
    ctk.CTkLabel(padre, text="Integración Numérica (Newton-Cotes)",
                 font=("Arial", 20, "bold")).pack(pady=10)

    # 1. Configuración de Modo
    frame_modo = ctk.CTkFrame(padre, fg_color="transparent")
    frame_modo.pack(pady=5)
    
    ctk.CTkLabel(frame_modo, text="Modo de Entrada:").grid(row=0, column=0, padx=5)
    combo_fuente = ctk.CTkOptionMenu(frame_modo, values=["Función f(x)", "Datos Tabulados"],
                                     command=lambda m: cambiar_modo(m))
    combo_fuente.grid(row=0, column=1, padx=5)

    ctk.CTkLabel(frame_modo, text="Método:").grid(row=0, column=2, padx=5)
    combo_metodo = ctk.CTkOptionMenu(frame_modo, values=["Trapecio Múltiple", "Simpson 1/3", "Simpson 3/8", "Trapecio Simple"])
    combo_metodo.grid(row=0, column=3, padx=5)

    # Contenedor dinámico (Función vs Tabla)
    frame_dinamico = ctk.CTkFrame(padre, fg_color="transparent")
    frame_dinamico.pack(pady=5, fill="x")

    # === RESULTADOS ===
    # Frame Scrollable para logs y gráfica
    frame_res = ctk.CTkScrollableFrame(padre, height=500, label_text="Resultados y Pasos")
    frame_res.pack(fill="both", expand=True, padx=10, pady=5)
    
    lbl_res_header = ctk.CTkLabel(frame_res, text="", font=("Arial", 16, "bold"), anchor="w")
    lbl_res_header.pack(fill="x", pady=5)
    
    lbl_res_body = ctk.CTkLabel(frame_res, text="", font=("Consolas", 12), justify="left", anchor="w")
    lbl_res_body.pack(fill="x", pady=5)
    
    # Contenedor para la gráfica DENTRO del scrollable
    frame_graf_interno = ctk.CTkFrame(frame_res, fg_color="transparent")
    frame_graf_interno.pack(fill="both", expand=True, pady=10)

    def limpiar_resultados():
        lbl_res_header.configure(text="")
        lbl_res_body.configure(text="Calculando...")
        for w in frame_graf_interno.winfo_children(): w.destroy()

    def graficar(x_suave, y_suave, x_ptos, y_ptos, titulo):
        for w in frame_graf_interno.winfo_children(): w.destroy()
        
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Curva
        ax.plot(x_suave, y_suave, 'b-', label='Function')
        ax.fill_between(x_suave, y_suave, alpha=0.2, color='skyblue')
        
        # Puntos iteración
        ax.plot(x_ptos, y_ptos, 'ro', markersize=4, label='Puntos')
        for xi in x_ptos:
            ax.axvline(x=xi, color='gray', linestyle=':', alpha=0.5)
            
        ax.set_title(f"Integral - {titulo}")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        canvas = FigureCanvasTkAgg(fig, master=frame_graf_interno)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Ya no necesitamos expandir la ventana principal dinámicamente si está en scroll
        # estado_graficas["Integración"] = True 
        # root.geometry(f"{ANCHO_APP}x{ALTO_GRAFICA}")

    # === LÓGICA DE CÁLCULO ===
    
    # Referencias globales para widgets (usando diccionario)
    widgets_ref = {}

    def obtener_parametros_funcion():
        fx = widgets_ref["fx"].get()
        a = float(widgets_ref["a"].get())
        b = float(widgets_ref["b"].get())
        
        modo_paso = widgets_ref["modo_paso"].get()
        val_paso = float(widgets_ref["val_paso"].get())
        
        if modo_paso == "Subintervalos (n)":
            n = int(val_paso)
        else:
            h = val_paso
            if h <= 0: raise ValueError("h debe ser > 0")
            n = int(round((b - a) / h))
            
        return fx, a, b, n

    def calcular_funcion():
        try:
            limpiar_resultados()
            fx, a, b, n = obtener_parametros_funcion()
            metodo = combo_metodo.get()
            
            # Ajustes para métodos específicos si n no es válido
            if metodo == "Trapecio Simple": n = 1
            if metodo == "Simpson 1/3" and n % 2 != 0: raise ValueError(f"Simpson 1/3 requiere n par. n={n} no es válido.")
            if metodo == "Simpson 3/8" and n % 3 != 0: raise ValueError(f"Simpson 3/8 requiere n múltiplo de 3. n={n} no es válido.")

            # Llamada API
            if metodo == "Trapecio Simple":
                res, h, xv, yv, xs, ys, pasos = calc.trapecio_simple(fx, a, b)
            elif metodo == "Trapecio Múltiple":
                res, h, xv, yv, xs, ys, pasos = calc.trapecio_multiple(fx, a, b, n)
            elif metodo == "Simpson 1/3":
                res, h, xv, yv, xs, ys, pasos = calc.simpson_1_3(fx, a, b, n)
            elif metodo == "Simpson 3/8":
                res, h, xv, yv, xs, ys, pasos = calc.simpson_3_8(fx, a, b, n)

            # Mostrar
            lbl_res_header.configure(text=f"Resultado: I ≈ {res:.8f}")
            lbl_res_body.configure(text=pasos)
            
            # Graficar
            graficar(xs, ys, xv, yv, metodo)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def analizar_convergencia():
        try:
            limpiar_resultados()
            fx, a, b, n = obtener_parametros_funcion()
            metodo = combo_metodo.get()
            
            # Si el método es simple, no tiene convergencia (n=1 fijo)
            if metodo == "Trapecio Simple": raise ValueError("Trapecio Simple (n=1) no permite análisis de convergencia.")
            
            # Valor real
            v_real = None
            if widgets_ref["real"].get().strip():
                v_real = float(widgets_ref["real"].get())
            
            reporte = calc.analisis_convergencia(fx, a, b, n, metodo, v_real)
            
            lbl_res_header.configure(text="Análisis de Convergencia")
            lbl_res_body.configure(text=reporte)
            
        except Exception as e:
             messagebox.showerror("Error", str(e))

    def calcular_tabla():
        try:
            limpiar_resultados()
            x_str = widgets_ref["tab_x"].get()
            y_str = widgets_ref["tab_y"].get()
            metodo = combo_metodo.get()
            
            res, x, y, pasos = calc.integrar_tabular(x_str, y_str, metodo)
            
            lbl_res_header.configure(text=f"Resultado: I ≈ {res:.8f}")
            lbl_res_body.configure(text=pasos)
            
            # Graficar (interpolación lineal visual)
            graficar(x, y, x, y, metodo + " (Datos)")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en cálculo tabular: {e}")

    # === CONFIGURACIÓN DE INPUTS ===
    # Definimos las funciones de UI que usan las funciones de cálculo aquí, DESPUÉS de definirlas

    def actualizar_input_paso(seleccion):
        pass 

    def configurar_inputs_funcion(frame):
        # Función
        ctk.CTkLabel(frame, text="Función f(x):").grid(row=0, column=0, padx=5, sticky="e")
        widgets_ref["fx"] = ctk.CTkEntry(frame, width=300, placeholder_text="Ej: x**2 + 2*x + 1")
        widgets_ref["fx"].grid(row=0, column=1, columnspan=3, padx=5, pady=5)

        # Límites
        ctk.CTkLabel(frame, text="a (inferior):").grid(row=1, column=0, padx=5, sticky="e")
        widgets_ref["a"] = ctk.CTkEntry(frame, width=100)
        widgets_ref["a"].grid(row=1, column=1, padx=5, sticky="w")

        ctk.CTkLabel(frame, text="b (superior):").grid(row=1, column=2, padx=5, sticky="e")
        widgets_ref["b"] = ctk.CTkEntry(frame, width=100)
        widgets_ref["b"].grid(row=1, column=3, padx=5, sticky="w")

        # Configuración de paso
        ctk.CTkLabel(frame, text="Definir por:").grid(row=2, column=0, padx=5, sticky="e")
        widgets_ref["modo_paso"] = ctk.CTkOptionMenu(frame, values=["Subintervalos (n)", "Tamaño de paso (h)"],
                                                     command=actualizar_input_paso)
        widgets_ref["modo_paso"].grid(row=2, column=1, padx=5)
        
        widgets_ref["val_paso"] = ctk.CTkEntry(frame, width=100, placeholder_text="Valor n o h")
        widgets_ref["val_paso"].grid(row=2, column=2, padx=5, sticky="w")

        # Valor Real (para convergencia)
        ctk.CTkLabel(frame, text="Valor Real (Opcional):").grid(row=3, column=0, padx=5, sticky="e")
        widgets_ref["real"] = ctk.CTkEntry(frame, width=150, placeholder_text="Para calcular error real")
        widgets_ref["real"].grid(row=3, column=1, columnspan=2, padx=5, sticky="w")
        
        # Botones de Acción
        frame_btns = ctk.CTkFrame(frame, fg_color="transparent")
        frame_btns.grid(row=4, column=0, columnspan=4, pady=10)
        
        ctk.CTkButton(frame_btns, text="Calcular Integral", command=calcular_funcion).pack(side="left", padx=5)
        ctk.CTkButton(frame_btns, text="Análisis de Convergencia", 
                      command=analizar_convergencia, fg_color="purple").pack(side="left", padx=5)
        ctk.CTkButton(frame_btns, text="Estimar Error (Cotas)", 
                      command=estimar_error, fg_color="#D35400").pack(side="left", padx=5)

    def estimar_error():
        try:
            limpiar_resultados()
            fx, a, b, n = obtener_parametros_funcion()
            metodo = combo_metodo.get()
            
            reporte = calc.estimar_error_cotas(fx, a, b, n, metodo)
            
            lbl_res_header.configure(text="Estimación de Error (Cotas Clásicas)")
            lbl_res_body.configure(text=reporte)
            
        except Exception as e:
             messagebox.showerror("Error", str(e))

    def configurar_inputs_tabla(frame):
        ctk.CTkLabel(frame, text="Datos X (comas):").grid(row=0, column=0, padx=5, sticky="e")
        widgets_ref["tab_x"] = ctk.CTkEntry(frame, width=400, placeholder_text="1, 1.5, 2.0, 2.5")
        widgets_ref["tab_x"].grid(row=0, column=1, padx=5, pady=5)

        ctk.CTkLabel(frame, text="Datos Y (comas):").grid(row=1, column=0, padx=5, sticky="e")
        widgets_ref["tab_y"] = ctk.CTkEntry(frame, width=400, placeholder_text="2.5, 3.1, 4.0, 5.2")
        widgets_ref["tab_y"].grid(row=1, column=1, padx=5, pady=5)
        
        ctk.CTkButton(frame, text="Calcular Integral Tabular", command=calcular_tabla).grid(row=2, column=0, columnspan=2, pady=10)

    def cambiar_modo(modo):
        for w in frame_dinamico.winfo_children(): w.destroy()
        
        if modo == "Función f(x)":
            configurar_inputs_funcion(frame_dinamico)
        else:
            configurar_inputs_tabla(frame_dinamico)

    # Nota: Inicializar el modo AL FINAL, después de que todas las funciones estén definidas
    # Inicializar modo
    configurar_inputs_funcion(frame_dinamico)


def configurar_pestana_derivacion(padre):
    ctk.CTkLabel(padre, text="Derivación Numérica",
                 font=("Arial", 20, "bold")).pack(pady=10)

    # Frame Configuración
    frame_config = ctk.CTkFrame(padre, fg_color="transparent")
    frame_config.pack(pady=5)

    ctk.CTkLabel(frame_config, text="Método:").grid(row=0, column=0, padx=5, sticky="e")
    combo_metodo = ctk.CTkOptionMenu(frame_config, values=["Adelante (Forward)", "Atrás (Backward)", "Centrada (Centered)", "Irregular (3 Puntos)"],
                                     command=lambda _: actualizar_entradas())
    combo_metodo.grid(row=0, column=1, padx=5, pady=5)
    
    ctk.CTkLabel(frame_config, text="Orden Error:").grid(row=0, column=2, padx=5, sticky="e")
    combo_orden = ctk.CTkOptionMenu(frame_config, values=["O(h)", "O(h^2)", "O(h^4)"])
    combo_orden.grid(row=0, column=3, padx=5, pady=5)

    # Frame Entradas
    frame_entradas = ctk.CTkFrame(padre, fg_color="transparent")
    frame_entradas.pack(pady=5)

    # Widgets para métodos regulares
    widgets_ref = {}
    
    lbl_fx = ctk.CTkLabel(frame_entradas, text="Función f(x):")
    widgets_ref["fx"] = ctk.CTkEntry(frame_entradas, width=250, placeholder_text="Ej: x**3 + 2*x")
    
    lbl_h = ctk.CTkLabel(frame_entradas, text="Paso (h):")
    widgets_ref["h"] = ctk.CTkEntry(frame_entradas, width=100, placeholder_text="0.1")

    # Widgets para método irregular
    lbl_datos_x = ctk.CTkLabel(frame_entradas, text="Datos X (comas):")
    widgets_ref["datos_x"] = ctk.CTkEntry(frame_entradas, width=250, placeholder_text="Ej: 1.2, 1.4, 1.6")
    
    lbl_datos_y = ctk.CTkLabel(frame_entradas, text="Datos Y (comas):")
    widgets_ref["datos_y"] = ctk.CTkEntry(frame_entradas, width=250, placeholder_text="Ej: 3.5, 4.1, 5.0")

    # Widget común
    lbl_x = ctk.CTkLabel(frame_entradas, text="Evaluar en x:")
    widgets_ref["x"] = ctk.CTkEntry(frame_entradas, width=100, placeholder_text="1.5")
    
    lbl_real = ctk.CTkLabel(frame_entradas, text="Valor Real (opcional):")
    widgets_ref["real"] = ctk.CTkEntry(frame_entradas, width=100)

    def actualizar_entradas():
        # Limpiar grid
        for w in [lbl_fx, widgets_ref["fx"], lbl_h, widgets_ref["h"], 
                  lbl_datos_x, widgets_ref["datos_x"], lbl_datos_y, widgets_ref["datos_y"], 
                  lbl_x, widgets_ref["x"], lbl_real, widgets_ref["real"]]:
            w.grid_forget()
        
        metodo = combo_metodo.get()
        
        # Ajustar opciones de orden según método
        if metodo == "Irregular (3 Puntos)":
            combo_orden.configure(state="disabled")
            
            lbl_datos_x.grid(row=0, column=0, padx=5, pady=5, sticky="e")
            widgets_ref["datos_x"].grid(row=0, column=1, padx=5, pady=5)
            lbl_datos_y.grid(row=1, column=0, padx=5, pady=5, sticky="e")
            widgets_ref["datos_y"].grid(row=1, column=1, padx=5, pady=5)
            lbl_x.grid(row=2, column=0, padx=5, pady=5, sticky="e")
            widgets_ref["x"].grid(row=2, column=1, padx=5, pady=5)
            lbl_real.grid(row=3, column=0, padx=5, pady=5, sticky="e")
            widgets_ref["real"].grid(row=3, column=1, padx=5, pady=5)
        else:
            combo_orden.configure(state="normal")
            vals_orden = ["O(h)", "O(h^2)"]
            if metodo == "Centrada (Centered)":
                vals_orden = ["O(h^2)", "O(h^4)"]
            combo_orden.configure(values=vals_orden)
            if combo_orden.get() not in vals_orden:
                combo_orden.set(vals_orden[0])

            lbl_fx.grid(row=0, column=0, padx=5, pady=5, sticky="e")
            widgets_ref["fx"].grid(row=0, column=1, padx=5, pady=5)
            lbl_x.grid(row=1, column=0, padx=5, pady=5, sticky="e")
            widgets_ref["x"].grid(row=1, column=1, padx=5, pady=5)
            lbl_h.grid(row=2, column=0, padx=5, pady=5, sticky="e")
            widgets_ref["h"].grid(row=2, column=1, padx=5, pady=5)
            lbl_real.grid(row=3, column=0, padx=5, pady=5, sticky="e")
            widgets_ref["real"].grid(row=3, column=1, padx=5, pady=5)

    actualizar_entradas()

    # Botones (Se crean primero el frame para que salga arriba)
    frame_btns = ctk.CTkFrame(padre, fg_color="transparent")
    frame_btns.pack(pady=5)

    # Frame Resultados (Scrollable)
    frame_res = ctk.CTkScrollableFrame(padre, height=400, label_text="Pasos y Resultados")
    frame_res.pack(fill="both", expand=True, padx=10, pady=5)
    
    lbl_res_header = ctk.CTkLabel(frame_res, text="", font=("Arial", 16, "bold"), anchor="w")
    lbl_res_header.pack(fill="x", pady=(5,0))
    
    lbl_res_content = ctk.CTkLabel(frame_res, text="", font=("Consolas", 12), justify="left", anchor="w")
    lbl_res_content.pack(fill="x", pady=5)

    def calcular():
        try:
            lbl_res_header.configure(text="")
            lbl_res_content.configure(text="")
            
            metodo = combo_metodo.get()
            x_val = float(widgets_ref["x"].get())
            
            # Obtener valor real si existe
            v_real = None
            if widgets_ref["real"].get().strip():
                v_real = float(widgets_ref["real"].get())

            if metodo == "Irregular (3 Puntos)":
                res, pasos, _, _ = calc.derivada_irregular(widgets_ref["datos_x"].get(), widgets_ref["datos_y"].get(), x_val)
                texto_pasos = pasos
            else:
                func = widgets_ref["fx"].get()
                h = float(widgets_ref["h"].get())
                if h == 0: raise ValueError("h no puede ser 0")
                
                orden = combo_orden.get()
                res, texto_pasos = calc.derivada_finitas(func, x_val, h, metodo, orden)

            # Mostrar resultado principal
            header_txt = f"Resultado: f'({x_val}) ≈ {res:.8f}"
            
            if v_real is not None:
                err_abs = abs(v_real - res)
                err_rel = (err_abs / abs(v_real)) * 100 if v_real != 0 else 0
                header_txt += f" | Err Rel: {err_rel:.4f}%"
                
            lbl_res_header.configure(text=header_txt)
            lbl_res_content.configure(text=texto_pasos)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            
    def analizar_h():
        try:
            lbl_res_header.configure(text="Análisis de Sensibilidad a h")
            
            metodo = combo_metodo.get()
            if metodo == "Irregular (3 Puntos)":
                lbl_res_content.configure(text="El análisis de 'h' solo aplica para métodos de diferencias finitas (regulares).")
                return

            func = widgets_ref["fx"].get()
            x_val = float(widgets_ref["x"].get())
            h = float(widgets_ref["h"].get())
            orden = combo_orden.get()
            
            v_real = None
            if widgets_ref["real"].get().strip():
                v_real = float(widgets_ref["real"].get())
                
            reporte = calc.analisis_error_derivada(func, x_val, h, metodo, orden, v_real)
            lbl_res_content.configure(text=reporte)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    # Agregar botones al frame que ya está arriba
    ctk.CTkButton(frame_btns, text="Calcular Derivada", command=calcular).pack(side="left", padx=5)
    ctk.CTkButton(frame_btns, text="Analizar Paso (h)", command=analizar_h, fg_color="purple").pack(side="left", padx=5)


def configurar_pestana_edo(padre):
    ctk.CTkLabel(padre, text="Ecuaciones Diferenciales Ordinarias",
                 font=("Arial", 20, "bold")).pack(pady=10)

    # Frame Principal con 2 columnas (Config/Entradas vs Gráfica) sería ideal, 
    # pero seguiremos el flujo vertical para simplificar, usando scroll general si es necesario.
    
    # --- Configuración ---
    frame_config = ctk.CTkFrame(padre, fg_color="transparent")
    frame_config.pack(pady=5)
    
    ctk.CTkLabel(frame_config, text="Método:").grid(row=0, column=0, padx=5, sticky="e")
    combo_metodo = ctk.CTkOptionMenu(frame_config, values=["Euler", "Runge-Kutta 4 (RK4)"])
    combo_metodo.grid(row=0, column=1, padx=5, pady=5)
    
    ctk.CTkLabel(frame_config, text="Condición de Parada:").grid(row=1, column=0, padx=5, sticky="e")
    combo_parada = ctk.CTkOptionMenu(frame_config, values=["Hasta x final", "Iteraciones (n)"],
                                     command=lambda _: actualizar_entradas())
    combo_parada.grid(row=1, column=1, padx=5, pady=5)

    # --- Entradas ---
    frame_entradas = ctk.CTkFrame(padre, fg_color="transparent")
    frame_entradas.pack(pady=5)

    widgets_ref = {}

    def crear_entrada(row, label_text, key, placeholder, width=150):
        ctk.CTkLabel(frame_entradas, text=label_text).grid(row=row, column=0, padx=5, sticky="e")
        entry = ctk.CTkEntry(frame_entradas, width=width, placeholder_text=placeholder)
        entry.grid(row=row, column=1, padx=5, pady=2)
        widgets_ref[key] = entry

    crear_entrada(0, "dy/dx = f(x,y):", "fxy", "Ej: x - y + 2", 300)
    crear_entrada(1, "Solución Exacta y(x):", "exacta", "Opcional. Ej: x + 1 + exp(-x)", 300)
    
    # Fila 2: x0, y0
    frame_init = ctk.CTkFrame(frame_entradas, fg_color="transparent")
    frame_init.grid(row=2, column=0, columnspan=2, pady=2)
    
    ctk.CTkLabel(frame_init, text="x0:").pack(side="left")
    widgets_ref["x0"] = ctk.CTkEntry(frame_init, width=80, placeholder_text="0")
    widgets_ref["x0"].pack(side="left", padx=5)
    
    ctk.CTkLabel(frame_init, text="y0:").pack(side="left")
    widgets_ref["y0"] = ctk.CTkEntry(frame_init, width=80, placeholder_text="2")
    widgets_ref["y0"].pack(side="left", padx=5)
    
    # Fila 3: Paso
    ctk.CTkLabel(frame_entradas, text="Paso (h):").grid(row=3, column=0, padx=5, sticky="e")
    widgets_ref["h"] = ctk.CTkEntry(frame_entradas, width=100, placeholder_text="0.1")
    widgets_ref["h"].grid(row=3, column=1, padx=5, pady=2, sticky="w")
    
    # Fila 4: Parada Dinámica
    lbl_parada = ctk.CTkLabel(frame_entradas, text="x final:")
    lbl_parada.grid(row=4, column=0, padx=5, sticky="e")
    widgets_ref["parada"] = ctk.CTkEntry(frame_entradas, width=100)
    widgets_ref["parada"].grid(row=4, column=1, padx=5, pady=2, sticky="w")

    def actualizar_entradas():
        modo = combo_parada.get()
        if modo == "Hasta x final":
            lbl_parada.configure(text="x final:")
        else:
            lbl_parada.configure(text="Iteraciones (n):")

    # Botones de Acción
    frame_btns = ctk.CTkFrame(padre, fg_color="transparent")
    frame_btns.pack(pady=5)
    
    ctk.CTkButton(frame_btns, text="Calcular y Graficar", command=lambda: calcular()).pack(side="left", padx=5)
    ctk.CTkButton(frame_btns, text="Verificar Orden (h vs h/2)", command=lambda: verificar_orden(), fg_color="purple").pack(side="left", padx=5)

    # --- Resultados ---
    frame_resultados = ctk.CTkScrollableFrame(padre, height=500, label_text="Resultados")
    frame_resultados.pack(fill="both", expand=True, padx=10, pady=5)
    
    # 1. Resumen Texto
    lbl_resumen = ctk.CTkLabel(frame_resultados, text="", font=("Arial", 14, "bold"), text_color="#4CC9F0")
    lbl_resumen.pack(pady=5)
    
    # 2. Tabla Logs
    lbl_log_header = ctk.CTkLabel(frame_resultados, text="", font=("Consolas", 12, "bold"), justify="left")
    lbl_log_header.pack(pady=(5,0))
    
    lbl_log_content = ctk.CTkLabel(frame_resultados, text="", font=("Consolas", 12), justify="left")
    lbl_log_content.pack(pady=5)

    # 3. Gráfica (Al final)
    frame_grafica = ctk.CTkFrame(frame_resultados, fg_color="transparent", height=300)
    frame_grafica.pack(fill="x", expand=False, pady=10)

    estado_graficas["EDO Numérica"] = True

    def obtener_entradas():
        func = widgets_ref["fxy"].get()
        exacta = widgets_ref["exacta"].get().strip()
        x0 = float(widgets_ref["x0"].get())
        y0 = float(widgets_ref["y0"].get())
        h = float(widgets_ref["h"].get())
        val_parada = float(widgets_ref["parada"].get())
        
        modo = combo_parada.get()
        if modo == "Hasta x final":
            n = int(round((val_parada - x0) / h))
        else:
            n = int(val_parada)
            
        if n <= 0: raise ValueError("N debe ser > 0")
        
        return func, exacta, x0, y0, h, n

    def calcular():
        try:
            # Limpiar
            for w in frame_grafica.winfo_children(): w.destroy()
            lbl_resumen.configure(text="")
            lbl_log_header.configure(text="")
            lbl_log_content.configure(text="")
            
            func, exacta, x0, y0, h, n = obtener_entradas()
            metodo = combo_metodo.get()
            
            sol_exacta_arg = exacta if exacta else None
            
            if metodo == "Euler":
                x_vals, y_vals, logs = calc.metodo_euler(func, x0, y0, h, n, sol_exacta_arg)
            else:
                x_vals, y_vals, logs = calc.metodo_rk4(func, x0, y0, h, n, sol_exacta_arg)
            
            # --- Graficar ---
            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            # Numérica
            ax.plot(x_vals, y_vals, 'o--', color='blue', markersize=4, label=f"Numérica ({metodo})")
            
            # Exacta (si existe)
            if sol_exacta_arg:
                x_smooth = np.linspace(x_vals[0], x_vals[-1], 200)
                try:
                    y_smooth = [calc.evaluar_funcion(exacta, xi) for xi in x_smooth]
                    ax.plot(x_smooth, y_smooth, 'g-', alpha=0.7, label="Exacta")
                except:
                    pass # Si falla graficar exacta, no importa
            
            ax.set_title(f"Solución EDO: {metodo}")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)

            # --- Tabla ---
            # Construir header dinámico
            cols = ["Iter", "xi", "yi"]
            if metodo == "Euler":
                cols.append("f(xi,yi)")
            else:
                cols.extend(["k1", "k2", "k3", "k4"])
            
            cols.append("yi+1")
            
            if sol_exacta_arg:
                cols.extend(["Exacta", "Error Abs"])
                
            # Formatear Header
            header_str = " | ".join([f"{c:^10}" for c in cols])
            
            # Formatear Content
            content_str = ""
            for paso in logs:
                row_str = ""
                # Datos básicos
                row_str += f"{paso['iter']:^10} | {paso['x']:^10.4f} | {paso['y']:^10.6f} | "
                
                if metodo == "Euler":
                    row_str += f"{paso['f_xy']:^10.4f} | "
                else:
                    # RK4 kvals (podrían ser largos, truncamos o 4 decimales)
                    row_str += f"{paso['k1']:^10.4f} | {paso['k2']:^10.4f} | {paso['k3']:^10.4f} | {paso['k4']:^10.4f} | "
                
                row_str += f"{paso['y_new']:^10.6f}"
                
                if sol_exacta_arg:
                    val_ex = paso.get('exacta')
                    val_err = paso.get('error')
                    txt_ex = f"{val_ex:.6f}" if val_ex is not None else "-"
                    txt_err = f"{val_err:.2e}" if val_err is not None else "-"
                    
                    row_str += f" | {txt_ex:^10} | {txt_err:^10}"
                
                content_str += row_str + "\n"
            
            lbl_log_header.configure(text=header_str)
            lbl_log_content.configure(text=content_str)
            
            # Resumen final
            res_txt = f"y({x_vals[-1]:.4f}) ≈ {y_vals[-1]:.6f}"
            if sol_exacta_arg:
                 err_final = logs[-1].get('error')
                 if err_final is not None:
                     res_txt += f" | Error Final: {err_final:.4e}"
            lbl_resumen.configure(text=res_txt)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def verificar_orden():
        try:
            # Limpiar resultados solo texto
            lbl_resumen.configure(text="Verificando Orden...")
            lbl_log_header.configure(text="Reporte de Convergencia")
            lbl_log_content.configure(text="")
            for w in frame_grafica.winfo_children(): w.destroy()
            
            func, exacta, x0, y0, h, n = obtener_entradas()
            metodo = combo_metodo.get()
            
            if not exacta:
                lbl_resumen.configure(text="Error: Se requiere solución exacta.")
                return

            reporte = calc.verificar_orden_edo(func, x0, y0, h, n, metodo, exacta)
            lbl_log_content.configure(text=reporte)
            lbl_resumen.configure(text="Verificación Completada")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))


# Agregar pestañas al Primer Parcial
tab_gauss_simple = vista_p1.add("Gauss Simple")
tab_gauss = vista_p1.add("Gauss-Seidel")
tab_pivoteo = vista_p1.add("Pivoteo")
tab_biseccion = vista_p1.add("Bisección")
tab_newton = vista_p1.add("Newton-Raphson")
tab_secante = vista_p1.add("Secante")
tab_secante_mod = vista_p1.add("Secante Modificada")
tab_regresion = vista_p1.add("Regresión")
tab_regresion_poli = vista_p1.add("Regresión Polinomial")

# Agregar pestañas al Segundo Parcial
tab_interpolacion = vista_p2.add("Interpolación")
tab_integracion = vista_p2.add("Integración")
tab_derivacion = vista_p2.add("Derivación Numérica")
tab_edo = vista_p2.add("EDO Numérica")

# Configurar pestañas Primer Parcial
configurar_pestana_gauss_simple(tab_gauss_simple)
configurar_pestana_gauss(tab_gauss)
configurar_pestana_pivoteo(tab_pivoteo)
configurar_pestana_biseccion(tab_biseccion)
configurar_pestana_newton(tab_newton)
configurar_pestana_secante(tab_secante)
configurar_pestana_secante_mod(tab_secante_mod)
configurar_pestana_regresion(tab_regresion)
configurar_pestana_regresion_poli(tab_regresion_poli)

# Configurar pestañas Segundo Parcial
configurar_pestana_interpolacion(tab_interpolacion)
configurar_pestana_integracion(tab_integracion)
configurar_pestana_derivacion(tab_derivacion)
configurar_pestana_edo(tab_edo)

root.mainloop()
