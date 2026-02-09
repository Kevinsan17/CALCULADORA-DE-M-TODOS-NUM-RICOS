import numpy as np


# ======================================================
# UTILIDADES
# ======================================================


def evaluar_funcion(funcion_str, x_val):
    """Evalúa una función string usando Numpy."""
    contexto = {
        "x": x_val,
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "pi": np.pi,
        "e": np.e
    }
    return eval(funcion_str, contexto)

# ======================================================
# ÁLGEBRA LINEAL
# ======================================================


def gauss_simple(A_str, b_str):
    A = np.array([list(map(float, row.split(",")))
                 for row in A_str.split(";")])
    b = np.array(list(map(float, b_str.split(","))))
    n = len(b)

    # Eliminación hacia adelante
    for k in range(n-1):
        for i in range(k+1, n):
            if A[k, k] == 0:
                raise ValueError("Pivote cero detectado.")
            factor = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] -= factor * A[k, j]
            b[i] -= factor * b[k]

    # Sustitución hacia atrás
    x = np.zeros(n)
    x[n-1] = b[n-1] / A[n-1, n-1]
    for i in range(n-2, -1, -1):
        sum_ax = sum(A[i, j] * x[j] for j in range(i+1, n))
        x[i] = (b[i] - sum_ax) / A[i, i]

    return x


def gauss_seidel(A_str, b_str, x0_str, tol=1e-4, max_iter=50):
    A = np.array([list(map(float, row.split(",")))
                 for row in A_str.split(";")])
    b = np.array(list(map(float, b_str.split(","))))
    x = np.array(list(map(float, x0_str.split(","))))
    n = len(b)

    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s1) / A[i][i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x

    return x  # Retorna la última aproximación


def pivoteo_parcial(A_str, b_str):
    A = np.array([list(map(float, row.split(",")))
                 for row in A_str.split(";")])
    b = np.array(list(map(float, b_str.split(","))))
    return np.linalg.solve(A, b)


def verificar_diagonal_dominante(A_str):
    """
    Devuelve True si la matriz es segura, False si no lo es,
    y una lista de las filas que causan problemas.
    """
    try:
        A = np.array([list(map(float, row.split(",")))
                     for row in A_str.split(";")])
        n = len(A)
        filas_problematicas = []
        es_dominante = True

        for i in range(n):
            # Valor absoluto de la diagonal
            diagonal = abs(A[i][i])
            # Suma de los valores absolutos de los vecinos
            suma_vecinos = sum(abs(A[i][j]) for j in range(n) if j != i)

            if diagonal < suma_vecinos:
                es_dominante = False
                # Guardamos el número de fila (1-based)
                filas_problematicas.append(i + 1)

        return es_dominante, filas_problematicas
    except:
        return False, []

# ======================================================
# ECUACIONES NO LINEALES
# ======================================================


def biseccion(func_str, a, b, tol):
    f_a = evaluar_funcion(func_str, a)
    f_b = evaluar_funcion(func_str, b)

    if f_a * f_b >= 0:
        raise ValueError("f(a) y f(b) deben tener signos opuestos.")

    xr = 0
    for i in range(100):
        xr_old = xr
        xr = (a + b) / 2

        if abs(xr - xr_old) < tol:
            return xr, i+1

        if evaluar_funcion(func_str, a) * evaluar_funcion(func_str, xr) < 0:
            b = xr
        else:
            a = xr

    return xr, 100


def newton_raphson(func_str, dfunc_str, x0, tol):
    x = x0
    for i in range(100):
        fx = evaluar_funcion(func_str, x)
        dfx = evaluar_funcion(dfunc_str, x)

        if abs(dfx) < 1e-8:
            raise ValueError("Derivada cercana a 0.")

        x_new = x - fx/dfx
        if abs(x_new - x) < tol:
            return x_new, i+1
        x = x_new
    raise ValueError("No convergió.")


def secante(func_str, x0, x1, tol):
    xi_1, xi = x0, x1
    for i in range(100):
        f0 = evaluar_funcion(func_str, xi_1)
        f1 = evaluar_funcion(func_str, xi)

        if f1 == f0:
            raise ValueError("División por cero.")

        xi_new = xi - f1 * (xi - xi_1) / (f1 - f0)
        if abs(xi_new - xi) < tol:
            return xi_new
        xi_1, xi = xi, xi_new
    raise ValueError("No convergió.")


def secante_modificada(func_str, x0, delta, tol):
    x = x0
    for i in range(100):
        fx = evaluar_funcion(func_str, x)
        fx_delta = evaluar_funcion(func_str, x + delta * x)

        denominador = fx_delta - fx
        if abs(denominador) < 1e-10:
            raise ValueError("División por cero.")

        x_new = x - (delta * x * fx) / denominador
        if abs(x_new - x) < tol:
            return x_new, i+1
        x = x_new
    raise ValueError("No convergió.")

# ======================================================
# REGRESIONES
# ======================================================


def regresion_lineal(x_str, y_str):
    x = np.array(list(map(float, x_str.split(","))))
    y = np.array(list(map(float, y_str.split(","))))

    if len(x) != len(y):
        raise ValueError("Datos de distinta longitud.")

    m, b = np.polyfit(x, y, 1)
    y_pred = m*x + b
    r2 = 1 - np.sum((y - y_pred)**2)/np.sum((y - np.mean(y))**2)

    ecuacion = f"y = {m:.4f}x + {b:.4f}"
    return x, y, y_pred, ecuacion, r2


def regresion_polinomial(x_str, y_str, grado):
    x = np.array(list(map(float, x_str.split(","))))
    y = np.array(list(map(float, y_str.split(","))))

    if len(x) != len(y):
        raise ValueError("Datos de distinta longitud.")

    coefs = np.polyfit(x, y, grado)
    poli = np.poly1d(coefs)
    y_pred = poli(x)
    r2 = 1 - np.sum((y - y_pred)**2)/np.sum((y - np.mean(y))**2)

    # Formateo bonito del texto
    ecuacion_texto = "y = "
    for i, c in enumerate(coefs):
        exponente = grado - i
        signo = "-" if c < 0 else "+" if i > 0 else ""
        valor = abs(c)

        if exponente == 0:
            termino = f"{signo} {valor:.4f}"
        elif exponente == 1:
            termino = f"{signo} {valor:.4f}x"
        else:
            termino = f"{signo} {valor:.4f}x^{exponente}"

        ecuacion_texto += termino + " "

    # Generar datos suaves para la gráfica
    x_suave = np.linspace(min(x), max(x), 100)
    y_suave = poli(x_suave)

    return x, y, x_suave, y_suave, ecuacion_texto, r2, grado


def interpolacion_lagrange(x_str, y_str, x_interp, x_val_str="", y_val_str=""):
    """
    Interpolación de Lagrange
    Retorna: x_datos, y_datos, x_eval, y_eval, polinomio_texto, y_interp, pasos_str, error_str, datos_validacion
    """
    x = np.array(list(map(float, x_str.split(","))))
    y = np.array(list(map(float, y_str.split(","))))

    if len(x) != len(y):
        raise ValueError("Los datos X e Y deben tener la misma longitud.")

    n = len(x)

    # Construir el polinomio completo y los pasos paso a paso
    polinomio = np.poly1d([0.0])
    pasos = []

    for k in range(n):
        # Construir L_k(x) numérico
        numerador = np.poly1d([1.0])
        denominador = 1.0
        
        termino_str_num = ""
        termino_str_den = ""

        for j in range(n):
            if j != k:
                numerador *= np.poly1d([1.0, -x[j]])
                denominador *= (x[k] - x[j])
                
                # Construcción de string para el paso
                signo = "-" if x[j] >= 0 else "+"
                val = abs(x[j])
                termino_str_num += f"(x {signo} {val:.2f})"
                
        termino_lagrange = (numerador / denominador) * y[k]
        polinomio += termino_lagrange
        
        # Guardar paso
        paso_k = f"L_{k}(x) = [{termino_str_num} / {denominador:.4f}] * {y[k]:.4f}"
        pasos.append(paso_k)

    # Evaluar usando el objeto polinomio
    y_interp = polinomio(x_interp)

    # Generar curva suave para graficar
    x_min, x_max = min(x), max(x)
    rango = x_max - x_min if x_max != x_min else 1.0
    x_eval = np.linspace(x_min - 0.1 * rango, x_max + 0.1 * rango, 200)
    y_eval = polinomio(x_eval)

    # Construir texto de la ecuación
    coefs = polinomio.coeffs
    grado = len(coefs) - 1
    ecuacion_texto = "P(x) ="
    
    primero = True
    for i, c in enumerate(coefs):
        if abs(c) < 1e-10:  # Ignorar coeficientes casi cero
            continue
            
        exponente = grado - i
        val_abs = abs(c)
        
        # Determinar signo y espaciado
        if primero:
            signo = "-" if c < 0 else ""
            primero = False
        else:
            signo = " - " if c < 0 else " + "
            
        # Formatear término
        if exponente == 0:
            termino = f"{signo}{val_abs:.4f}"
        elif exponente == 1:
            termino = f"{signo}{val_abs:.4f}x"
        else:
            termino = f"{signo}{val_abs:.4f}x^{exponente}"
            
        ecuacion_texto += termino

    if primero: # Si todos fueron cero
        ecuacion_texto += " 0"

    polinomio_texto = f"{ecuacion_texto}\nResultado: P({x_interp}) = {y_interp:.6f}"
    
    pasos_str = "\n".join(pasos)

    # Cálculo de errores si hay datos de validación
    error_str = "No se proporcionaron datos de validación."
    val_data = None
    
    if x_val_str and y_val_str:
        try:
            xv = np.array(list(map(float, x_val_str.split(","))))
            yv = np.array(list(map(float, y_val_str.split(","))))
            
            if len(xv) != len(yv):
                error_str = "Error: Datos de validación X e Y de distinta longitud."
            else:
                yv_pred = polinomio(xv)
                errores = np.abs(yv - yv_pred)
                
                max_err = np.max(errores)
                rms_err = np.sqrt(np.mean(errores**2))
                
                error_str = f"Análisis de Error (Validación):\nError Máximo: {max_err:.6f}\nError RMS: {rms_err:.6f}"
                val_data = (xv, yv)
        except Exception as e:
            error_str = f"Error al procesar validación: {str(e)}"

    return x, y, x_eval, y_eval, polinomio_texto, y_interp, pasos_str, error_str, val_data


def interpolacion_newton(x_str, y_str, x_interp):
    """
    Interpolación de Newton con diferencias divididas
    Retorna: x_datos, y_datos, x_eval, y_eval, polinomio_texto, tabla_diferencias
    """
    x = np.array(list(map(float, x_str.split(","))))
    y = np.array(list(map(float, y_str.split(","))))

    if len(x) != len(y):
        raise ValueError("Los datos X e Y deben tener la misma longitud.")

    n = len(x)

    # Construir tabla de diferencias divididas
    tabla = np.zeros((n, n))
    tabla[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            tabla[i][j] = (tabla[i + 1][j - 1] - tabla[i][j - 1]) / (x[i + j] - x[i])

    # Coeficientes son la primera fila
    coef = tabla[0, :]

    # Construir el polinomio completo usando np.poly1d para obtener la ecuación expandida
    polinomio = np.poly1d([0.0])
    
    for i in range(n):
        # Término b_i * (x-x0) * ... * (x-x_{i-1})
        termino = np.poly1d([coef[i]])
        for k in range(i):
            termino *= np.poly1d([1.0, -x[k]])
        polinomio += termino

    # Evaluar usando el objeto polinomio
    y_interp = polinomio(x_interp)

    # Generar curva suave para graficar
    x_min, x_max = min(x), max(x)
    rango = x_max - x_min if x_max != x_min else 1.0
    x_eval = np.linspace(x_min - 0.1 * rango, x_max + 0.1 * rango, 200)
    y_eval = polinomio(x_eval)

    # Construir texto de la ecuación
    coefs_poly = polinomio.coeffs
    grado = len(coefs_poly) - 1
    ecuacion_texto = "P(x) ="
    
    primero = True
    for i, c in enumerate(coefs_poly):
        if abs(c) < 1e-10:
            continue
            
        exponente = grado - i
        val_abs = abs(c)
        
        if primero:
            signo = "-" if c < 0 else ""
            primero = False
        else:
            signo = " - " if c < 0 else " + "
            
        if exponente == 0:
            termino = f"{signo}{val_abs:.4f}"
        elif exponente == 1:
            termino = f"{signo}{val_abs:.4f}x"
        else:
            termino = f"{signo}{val_abs:.4f}x^{exponente}"
            
        ecuacion_texto += termino

    if primero:
        ecuacion_texto += " 0"

    polinomio_texto = f"{ecuacion_texto}\nResultado: P({x_interp}) = {y_interp:.6f}"

    return x, y, x_eval, y_eval, polinomio_texto, y_interp


def trapecio_simple(func_str, a, b):
    """
    Método del Trapecio Simple (n=1)
    """
    h = b - a
    fa = evaluar_funcion(func_str, a)
    fb = evaluar_funcion(func_str, b)
    
    resultado = (h / 2) * (fa + fb)
    
    # Logs Detallados
    pasos = f"--- Trapecio Simple ---\n"
    pasos += f"Intervalo [{a}, {b}], h = {h:.6f}\n"
    pasos += f"f(a) = {fa:.6f}\n"
    pasos += f"f(b) = {fb:.6f}\n"
    pasos += f"I ≈ (h/2) * [f(a) + f(b)]\n"
    pasos += f"I ≈ ({h:.6f}/2) * [{fa:.6f} + {fb:.6f}]\n"
    pasos += f"I ≈ {resultado:.8f}"
    
    x_vals = np.array([a, b])
    y_vals = np.array([fa, fb])
    x_suave = np.linspace(a, b, 100)
    y_suave = np.array([evaluar_funcion(func_str, xi) for xi in x_suave])
    
    return resultado, h, x_vals, y_vals, x_suave, y_suave, pasos


def trapecio_multiple(func_str, a, b, n):
    """
    Método del Trapecio Múltiple
    """
    if n < 1: raise ValueError("n >= 1")
    h = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)
    y_vals = np.array([evaluar_funcion(func_str, x) for x in x_vals])
    
    suma_interna = np.sum(y_vals[1:-1])
    suma_total = y_vals[0] + 2 * suma_interna + y_vals[-1]
    resultado = (h / 2) * suma_total
    
    pasos = f"--- Trapecio Múltiple (n={n}) ---\n"
    pasos += f"h = ({b} - {a}) / {n} = {h:.6f}\n\n"
    pasos += f"Puntos: {x_vals}\n"
    pasos += f"Valores f(x): {y_vals}\n\n"
    pasos += f"Suma extremos: f(x0) + f(xn) = {y_vals[0]:.6f} + {y_vals[-1]:.6f} = {y_vals[0]+y_vals[-1]:.6f}\n"
    pasos += f"Suma internos (x1...xn-1): {suma_interna:.6f}\n"
    pasos += f"Fórmula: (h/2) * [ f(x0) + 2*Sum_int + f(xn) ]\n"
    pasos += f"I ≈ ({h:.6f}/2) * [ {y_vals[0]+y_vals[-1]:.6f} + 2*({suma_interna:.6f}) ]\n"
    pasos += f"I ≈ {resultado:.8f}"

    x_suave = np.linspace(a, b, 200)
    y_suave = np.array([evaluar_funcion(func_str, xi) for xi in x_suave])
    
    return resultado, h, x_vals, y_vals, x_suave, y_suave, pasos


def simpson_1_3(func_str, a, b, n):
    """
    Regla de Simpson 1/3 (n par)
    """
    if n % 2 != 0: raise ValueError("n debe ser par")
    h = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)
    y_vals = np.array([evaluar_funcion(func_str, x) for x in x_vals])

    suma_impares = np.sum(y_vals[1:-1:2])
    suma_pares = np.sum(y_vals[2:-1:2])
    
    suma_total = y_vals[0] + 4 * suma_impares + 2 * suma_pares + y_vals[-1]
    resultado = (h / 3) * suma_total

    pasos = f"--- Simpson 1/3 (n={n}) ---\n"
    pasos += f"h = {h:.6f}\n\n"
    pasos += f"Extremos: f(x0)={y_vals[0]:.6f}, f(xn)={y_vals[-1]:.6f}\n"
    pasos += f"Suma impares (coef 4): {suma_impares:.6f}\n"
    pasos += f"Suma pares (coef 2): {suma_pares:.6f}\n"
    pasos += f"Fórmula: (h/3) * [ f(x0) + 4*Impares + 2*Pares + f(xn) ]\n"
    pasos += f"I ≈ ({h:.6f}/3) * [ {y_vals[0]+y_vals[-1]:.6f} + 4*({suma_impares:.6f}) + 2*({suma_pares:.6f}) ]\n"
    pasos += f"I ≈ {resultado:.8f}"

    x_suave = np.linspace(a, b, 200)
    y_suave = np.array([evaluar_funcion(func_str, xi) for xi in x_suave])

    return resultado, h, x_vals, y_vals, x_suave, y_suave, pasos


def simpson_3_8(func_str, a, b, n):
    """
    Regla de Simpson 3/8 (n múltiplo de 3)
    """
    if n % 3 != 0: raise ValueError("n debe ser múltiplo de 3")
    h = (b - a) / n
    x_vals = np.linspace(a, b, n + 1)
    y_vals = np.array([evaluar_funcion(func_str, x) for x in x_vals])

    suma_multiplos_3 = 0
    suma_resto = 0
    
    for i in range(1, n):
        if i % 3 == 0:
            suma_multiplos_3 += y_vals[i]
        else:
            suma_resto += y_vals[i]

    suma_total = y_vals[0] + 3 * suma_resto + 2 * suma_multiplos_3 + y_vals[-1]
    resultado = (3 * h / 8) * suma_total

    pasos = f"--- Simpson 3/8 (n={n}) ---\n"
    pasos += f"h = {h:.6f}\n\n"
    pasos += f"Extremos: {y_vals[0]:.6f} + {y_vals[-1]:.6f}\n"
    pasos += f"Suma resto (coef 3): {suma_resto:.6f}\n"
    pasos += f"Suma múlt 3 (coef 2): {suma_multiplos_3:.6f}\n"
    pasos += f"Fórmula: (3h/8) * [ f(x0) + 3*Resto + 2*Mult3 + f(xn) ]\n"
    pasos += f"I ≈ (3*{h:.6f}/8) * [ {y_vals[0]+y_vals[-1]:.6f} + 3*({suma_resto:.6f}) + 2*({suma_multiplos_3:.6f}) ]\n"
    pasos += f"I ≈ {resultado:.8f}"

    x_suave = np.linspace(a, b, 200)
    y_suave = np.array([evaluar_funcion(func_str, xi) for xi in x_suave])

    return resultado, h, x_vals, y_vals, x_suave, y_suave, pasos


def integrar_tabular(x_str, y_str, metodo):
    """
    Integración para datos tabulados
    """
    x = np.array(list(map(float, x_str.split(","))))
    y = np.array(list(map(float, y_str.split(","))))

    if len(x) != len(y): raise ValueError("Longitud de X e Y no coincide")
    n = len(x) - 1
    if n < 1: raise ValueError("Insuficientes datos")

    h_arr = np.diff(x)
    h_prom = np.mean(h_arr)
    es_uniforme = np.allclose(h_arr, h_prom, atol=1e-5)
    
    pasos = f"Datos: {len(x)} puntos (n={n})\n"

    if metodo == "Trapecio Múltiple":
        # Trapecio funciona con paso variable o fijo
        # Fórmula general: sum( (xi+1 - xi) * (yi + yi+1) / 2 )
        areas = h_arr * (y[:-1] + y[1:]) / 2
        resultado = np.sum(areas)
        pasos += f"Paso variable/fijo. Suma de áreas trapezoidales.\n"
        
    elif "Simpson" in metodo:
        if not es_uniforme:
            raise ValueError("Simpson requiere paso uniforme. Use Trapecio o regularice los datos.")
        
        h = h_prom
        pasos += f"Paso uniforme detectado h ≈ {h:.6f}\n"
        
        if metodo == "Simpson 1/3":
            if n % 2 != 0: raise ValueError("Simpson 1/3 requiere n par (número impar de puntos).")
            suma = y[0] + y[-1] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-1:2])
            resultado = (h/3) * suma
            
        elif metodo == "Simpson 3/8":
            if n % 3 != 0: raise ValueError("Simpson 3/8 requiere n múltiplo de 3.")
            suma_rest = 0
            suma_mult3 = 0
            for i in range(1, n):
                if i % 3 == 0: suma_mult3 += y[i]
                else: suma_rest += y[i]
            resultado = (3*h/8) * (y[0] + 3*suma_rest + 2*suma_mult3 + y[-1])
            
    else:
        raise ValueError("Método desconocido")
        
    pasos += f"Resultado Final I ≈ {resultado:.8f}"
    return resultado, x, y, pasos


def analisis_convergencia(func_str, a, b, n_inicial, metodo, valor_real=None):
    """
    Calcula la integral con n y 2n para estimar orden de convergencia.
    """
    # Función auxiliar para llamar al método correcto solo devolviendo resultado
    def calcular(n_local):
        if metodo == "Simpson 1/3" and n_local % 2 != 0: n_local += 1
        if metodo == "Simpson 3/8" and n_local % 3 != 0: n_local += (3 - (n_local % 3))
        
        if metodo == "Trapecio Múltiple":
            res, _, _, _, _, _, _ = trapecio_multiple(func_str, a, b, n_local)
        elif metodo == "Simpson 1/3":
            res, _, _, _, _, _, _ = simpson_1_3(func_str, a, b, n_local)
        elif metodo == "Simpson 3/8":
            res, _, _, _, _, _, _ = simpson_3_8(func_str, a, b, n_local)
        else:
            return 0, n_local
        return res, n_local

    # Caso 1: n
    i1, n1 = calcular(n_inicial)
    
    # Caso 2: 2n (Aproximadamente, ajustado a requisitos del método)
    i2, n2 = calcular(n_inicial * 2)

    reporte = f"--- Análisis de Convergencia ---\n"
    reporte += f"n1 = {n1} -> I1 = {i1:.8f}\n"
    reporte += f"n2 = {n2} -> I2 = {i2:.8f}\n"
    
    if valor_real is not None:
        e1 = abs(valor_real - i1)
        e2 = abs(valor_real - i2)
        reporte += f"Error1 = {e1:.8e}\nError2 = {e2:.8e}\n"
        
        if e2 > 0 and e1 > 0:
            # p ≈ log2(E1/E2) para h dividiéndose por 2 (n multiplicándose por 2)
            # Nota: si n2 no es exactamente 2*n1 (debido a ajustes Simpson), esto es una aprox
            ratio_n = n2 / n1
            ratio_error = e1 / e2
            # E ~ C * (1/n)^p  => E1/E2 ~ (n2/n1)^p
            # log(E1/E2) ~ p * log(n2/n1)
            p = np.log(ratio_error) / np.log(ratio_n)
            reporte += f"Orden Empírico p ≈ {p:.4f}\n"
            
            # Advertencia de suavidad si p está muy lejos del teórico
            teorico = 2 if "Trapecio" in metodo else 4
            if p < teorico - 1:
                reporte += "⚠️ Advertencia: El orden es bajo. La función podría no ser suave.\n"
        else:
            reporte += "Orden: No calculable (error cero o indefinido).\n"
    else:
        reporte += "Orden: Requiere Valor Real para calcular p.\n"
        
    return reporte


def estimar_error_cotas(func_str, a, b, n, metodo):
    """
    Estima el error usando cotas clásicas con derivadas numéricas máximas.
    Retorna string con el reporte.
    """
    import math

    if metodo not in ["Trapecio Simple", "Trapecio Múltiple", "Simpson 1/3", "Simpson 3/8"]:
        return "Método no aplicable para cotas de error clásicas."

    # Paso h
    h = (b - a) / n
    
    # Derivación numérica para encontrar max|f''(x)| o max|f''''(x)|
    # Usamos un barrido de puntos
    puntos_eval = np.linspace(a, b, 200)
    
    def diff_n(f_str, x, order, h_diff=1e-4):
        # Diferencias finitas centradas recursivas o simples
        # Para orden 2: (f(x+h) - 2f(x) + f(x-h)) / h^2
        # Para orden 4: Aproximación numérica
        if order == 2:
            return (evaluar_funcion(f_str, x + h_diff) - 2*evaluar_funcion(f_str, x) + evaluar_funcion(f_str, x - h_diff)) / (h_diff**2)
        elif order == 4:
            # Aprox orden 4 centrada
            # f''''(x) ≈ (f(x+2h) - 4f(x+h) + 6f(x) - 4f(x-h) + f(x-2h)) / h^4
            t1 = evaluar_funcion(f_str, x + 2*h_diff)
            t2 = -4 * evaluar_funcion(f_str, x + h_diff)
            t3 = 6 * evaluar_funcion(f_str, x)
            t4 = -4 * evaluar_funcion(f_str, x - h_diff)
            t5 = evaluar_funcion(f_str, x - 2*h_diff)
            return (t1 + t2 + t3 + t4 + t5) / (h_diff**4)
        return 0

    order_deriv = 2 if "Trapecio" in metodo else 4
    
    max_val_deriv = 0
    vals_deriv = []
    
    for val_x in puntos_eval:
        try:
            d = abs(diff_n(func_str, val_x, order_deriv))
            vals_deriv.append(d)
        except:
            pass
            
    if not vals_deriv:
        return "No se pudo evaluar derivadas en el intervalo."
        
    max_val_deriv = max(vals_deriv)
    
    # Calcular Cota
    error_cota = 0
    formula_str = ""
    
    if "Trapecio" in metodo:
        # |E| <= (b-a)^3 / (12n^2) * max|f''|
        numerador = (b - a)**3
        denominador = 12 * (n**2)
        error_cota = (numerador / denominador) * max_val_deriv
        formula_str = "|E| <= [(b-a)^3 / 12n^2] * max|f''(x)|"
        
    elif metodo == "Simpson 1/3":
        # |E| <= (b-a)^5 / (180n^4) * max|f''''(x)|
        numerador = (b - a)**5
        denominador = 180 * (n**4)
        error_cota = (numerador / denominador) * max_val_deriv
        formula_str = "|E| <= [(b-a)^5 / 180n^4] * max|f''''(x)|"
        
    elif metodo == "Simpson 3/8":
        # |E| <= (b-a)^5 / (80n^4) * max|f''''(x)|  -- Nota: a veces es 3h^5/80... usando forma estándar (b-a)
        # Standard: 3/80 * h^5 * f''''(xi) * num_intervalos? 
        # Fórmula global: |E| <= (b-a) * h^4 / 80 * max|f''''|
        # h = (b-a)/n -> h^4 = (b-a)^4 / n^4
        # |E| <= (b-a)^5 / (80 * n^4) * max
        numerador = (b - a)**5
        denominador = 80 * (n**4)
        error_cota = (numerador / denominador) * max_val_deriv
        formula_str = "|E| <= [(b-a)^5 / 80n^4] * max|f''''(x)|"

    reporte = f"--- Estimación de Error (Cotas) ---\n"
    reporte += f"Método: {metodo}\n"
    reporte += f"Orden Derivada requerida: {order_deriv}\n"
    reporte += f"Máx |f^({order_deriv})(x)| estimado en [{a}, {b}]: {max_val_deriv:.6f}\n"
    reporte += f"Fórmula: {formula_str}\n"
    reporte += f"Error estimado (Cota Superior): {error_cota:.8e}\n"
    
    return reporte


# ======================================================
# DERIVACIÓN NUMÉRICA
# ======================================================


def derivada_finitas(func_str, x, h, metodo, orden_error="O(h^2)"):
    """
    Calcula derivada usando diferencias finitas.
    Soporta: Adelante, Atrás, Centrada.
    Órdenes de error: O(h), O(h^2), O(h^4) según corresponda.
    """
    pasos = f"--- Derivación Numérica: {metodo} ---\n"
    pasos += f"f(x) = {func_str}\n"
    pasos += f"x0 = {x}, h = {h}\n\n"
    
    # Evaluar puntos necesarios
    # Se evalúan bajo demanda para optimizar logs
    
    fx = evaluating_wrapper(func_str, x)
    
    res = 0
    formula_str = ""
    vals_str = ""
    
    if metodo == "Adelante (Forward)":
        if orden_error == "O(h)":
            # f'(x) = (f(x+h) - f(x)) / h
            fxh = evaluating_wrapper(func_str, x + h)
            res = (fxh - fx) / h
            formula_str = "[f(x+h) - f(x)] / h"
            vals_str = f"[{fxh:.6f} - {fx:.6f}] / {h:.6f}"
            
        elif orden_error == "O(h^2)":
            # f'(x) = (-f(x+2h) + 4f(x+h) - 3f(x)) / 2h
            fxh = evaluating_wrapper(func_str, x + h)
            fx2h = evaluating_wrapper(func_str, x + 2*h)
            res = (-fx2h + 4*fxh - 3*fx) / (2*h)
            formula_str = "[-f(x+2h) + 4f(x+h) - 3f(x)] / 2h"
            vals_str = f"[-{fx2h:.6f} + 4({fxh:.6f}) - 3({fx:.6f})] / {2*h:.6f}"
            
    elif metodo == "Atrás (Backward)":
        if orden_error == "O(h)":
            # f'(x) = (f(x) - f(x-h)) / h
            fxmh = evaluating_wrapper(func_str, x - h)
            res = (fx - fxmh) / h
            formula_str = "[f(x) - f(x-h)] / h"
            vals_str = f"[{fx:.6f} - {fxmh:.6f}] / {h:.6f}"
            
        elif orden_error == "O(h^2)":
            # f'(x) = (3f(x) - 4f(x-h) + f(x-2h)) / 2h
            fxmh = evaluating_wrapper(func_str, x - h)
            fxm2h = evaluating_wrapper(func_str, x - 2*h)
            res = (3*fx - 4*fxmh + fxm2h) / (2*h)
            formula_str = "[3f(x) - 4f(x-h) + f(x-2h)] / 2h"
            vals_str = f"[3({fx:.6f}) - 4({fxmh:.6f}) + {fxm2h:.6f}] / {2*h:.6f}"

    elif metodo == "Centrada (Centered)":
        if orden_error == "O(h^2)":
            # f'(x) = (f(x+h) - f(x-h)) / 2h
            fxh = evaluating_wrapper(func_str, x + h)
            fxmh = evaluating_wrapper(func_str, x - h)
            res = (fxh - fxmh) / (2*h)
            formula_str = "[f(x+h) - f(x-h)] / 2h"
            vals_str = f"[{fxh:.6f} - {fxmh:.6f}] / {2*h:.6f}"
            
        elif orden_error == "O(h^4)":
            # f'(x) = (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / 12h
            fx2h = evaluating_wrapper(func_str, x + 2*h)
            fxh = evaluating_wrapper(func_str, x + h)
            fxmh = evaluating_wrapper(func_str, x - h)
            fxm2h = evaluating_wrapper(func_str, x - 2*h)
            res = (-fx2h + 8*fxh - 8*fxmh + fxm2h) / (12*h)
            formula_str = "[-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / 12h"
            vals_str = f"[-{fx2h:.6f} + 8({fxh:.6f}) - 8({fxmh:.6f}) + {fxm2h:.6f}] / {12*h:.6f}"
    
    pasos += f"Precisión: {orden_error}\n"
    pasos += f"Fórmula: {formula_str}\n"
    pasos += f"Sustitución: {vals_str}\n\n"
    pasos += f"Resultado: f'({x}) ≈ {res:.8f}"
    
    return res, pasos

def evaluating_wrapper(f_str, x):
    # Wrapper simple para uso interno
    return evaluar_funcion(f_str, x)

def derivada_irregular(x_arr, y_arr, x0):
    """
    Derivada usando polinomio de Lagrange de 3 puntos (datos irregulares).
    Busca los 3 puntos más cercanos a x0.
    """
    # 1. Encontrar los 3 índices más cercanos
    # Convertir a numpy array por si acaso vienen como listas
    x_arr = np.array(list(map(float, x_arr.split(",")))) if isinstance(x_arr, str) else np.array(x_arr)
    y_arr = np.array(list(map(float, y_arr.split(",")))) if isinstance(y_arr, str) else np.array(y_arr)

    if len(x_arr) < 3:
        raise ValueError("Se requieren al menos 3 puntos para derivada irregular.")

    diffs = np.abs(x_arr - float(x0))
    indices = np.argsort(diffs)[:3]
    # Ordenar por x para evitar confusiones en fórmula
    indices = np.sort(indices)
    
    x_p = x_arr[indices]
    y_p = y_arr[indices]
    
    x0_val = float(x0) # El punto donde evaluamos
    
    x_0, x_1, x_2 = x_p[0], x_p[1], x_p[2]
    y_0, y_1, y_2 = y_p[0], y_p[1], y_p[2]
    
    # Fórmula derivada Polinomio Lagrange 2do grado evaluada en x0_val
    # L0'(x) = (2x - x1 - x2) / ((x0-x1)(x0-x2))
    L0_p = (2*x0_val - x_1 - x_2) / ((x_0 - x_1)*(x_0 - x_2))
    
    # L1'(x) = (2x - x0 - x2) / ((x1-x0)(x1-x2))
    L1_p = (2*x0_val - x_0 - x_2) / ((x_1 - x_0)*(x_1 - x_2))
    
    # L2'(x) = (2x - x0 - x1) / ((x2-x0)(x2-x1))
    L2_p = (2*x0_val - x_0 - x_1) / ((x_2 - x_0)*(x_2 - x_1))
    
    derivada = y_0 * L0_p + y_1 * L1_p + y_2 * L2_p
    
    pasos = "--- Derivada Irregular (Lagrange 3 Puntos) ---\n"
    pasos += f"Puntos usados cercanos a x={x0_val}:\n"
    pasos += f"A({x_0}, {y_0}), B({x_1}, {y_1}), C({x_2}, {y_2})\n\n"
    pasos += f"Derivadas de Coeficientes Lagrange en x={x0_val}:\n"
    pasos += f"L0' = {L0_p:.6f}\n"
    pasos += f"L1' = {L1_p:.6f}\n"
    pasos += f"L2' = {L2_p:.6f}\n\n"
    pasos += f"f'(x) ≈ y0*L0' + y1*L1' + y2*L2'\n"
    pasos += f"f'(x) ≈ {y_0}*({L0_p:.4f}) + {y_1}*({L1_p:.4f}) + {y_2}*({L2_p:.4f})\n"
    pasos += f"Resultado ≈ {derivada:.8f}"
    
    return derivada, pasos, x_p, y_p


def analisis_error_derivada(func_str, x0, h_inicial, metodo, orden, valor_real=None):
    """
    Analiza cómo cambia la derivada variando h.
    Retorna reporte texto.
    """
    reporte = "--- Análisis de Sensibilidad a 'h' ---\n"
    reporte += f"Método: {metodo} | Orden: {orden}\n"
    reporte += f"Valor Real: {valor_real if valor_real is not None else 'No provisto'}\n\n"
    
    h_vals = [h_inicial, h_inicial/2, h_inicial/10, h_inicial/100, h_inicial/1000]
    
    reporte += f"{'h':<12} | {'Aprox':<12} | {'Error Abs':<12}\n"
    reporte += "-"*45 + "\n"
    
    for h in h_vals:
        try:
            val, _ = derivada_finitas(func_str, x0, h, metodo, orden)
            error_str = "N/A"
            if valor_real is not None:
                err = abs(val - valor_real)
                error_str = f"{err:.2e}"
            
            reporte += f"{h:<12.6f} | {val:<12.6f} | {error_str:<12}\n"
        except:
             reporte += f"{h:<12.6f} | {'Error Calc':<12} | {'-':<12}\n"
             
    if valor_real is not None:
        reporte += "\nNota: Si el error aumenta con h muy pequeño,\nindica error de redondeo/cancelación numérica."
        
    return reporte


# ======================================================
# ECUACIONES DIFERENCIALES ORDINARIAS (EDO)
# ======================================================

def evaluar_funcion_xy(funcion_str, x_val, y_val):
    """Evalúa una función f(x,y) de forma segura."""
    contexto = {
        "x": x_val,
        "y": y_val,
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "pi": np.pi,
        "e": np.e
    }
    return eval(funcion_str, contexto)


def metodo_euler(func_str, x0, y0, h, n, sol_exacta_str=None):
    """
    Método de Euler.
    Retorna: x_vals, y_vals, log_pasos
    """
    x_vals = [x0]
    y_vals = [y0]
    log_pasos = []

    xi = x0
    yi = y0

    # Valor inicial exacto y error
    exacta_0 = None
    error_0 = None
    if sol_exacta_str:
        try:
            exacta_0 = evaluar_funcion(sol_exacta_str, x0)
            error_0 = abs(yi - exacta_0)
        except: pass

    # Log inicial
    log_pasos.append({
        "iter": 0,
        "x": xi,
        "y": yi,
        "f_xy": 0,
        "y_new": yi,
        "exacta": exacta_0,
        "error": error_0
    })

    for i in range(n):
        # f(xi, yi)
        pendiente = evaluar_funcion_xy(func_str, xi, yi)
        
        # yi+1 = yi + h * pendiente
        yi_new = yi + h * pendiente
        xi_new = xi + h
        
        # Exacta y Error
        exacta_val = None
        error_val = None
        if sol_exacta_str:
            try:
                exacta_val = evaluar_funcion(sol_exacta_str, xi_new)
                error_val = abs(yi_new - exacta_val)
            except: pass

        detalle = {
            "iter": i+1,
            "x": xi,
            "y": yi,
            "f_xy": pendiente,
            "y_new": yi_new,
            "exacta": exacta_val,
            "error": error_val
        }
        log_pasos.append(detalle)
        
        xi = xi_new
        yi = yi_new
        
        x_vals.append(xi)
        y_vals.append(yi)

    return np.array(x_vals), np.array(y_vals), log_pasos


def metodo_rk4(func_str, x0, y0, h, n, sol_exacta_str=None):
    """
    Método de Runge-Kutta de 4to Orden (RK4).
    Retorna: x_vals, y_vals, log_pasos
    """
    x_vals = [x0]
    y_vals = [y0]
    log_pasos = []

    xi = x0
    yi = y0
    
    # Valor inicial exacto
    exacta_0 = None
    error_0 = None
    if sol_exacta_str:
        try:
            exacta_0 = evaluar_funcion(sol_exacta_str, x0)
            error_0 = abs(yi - exacta_0)
        except: pass

    log_pasos.append({
        "iter": 0,
        "x": xi,
        "y": yi,
        "k1": 0, "k2": 0, "k3": 0, "k4": 0,
        "y_new": yi,
        "exacta": exacta_0,
        "error": error_0
    })

    for i in range(n):
        # Cálculo de k1, k2, k3, k4
        k1 = evaluar_funcion_xy(func_str, xi, yi)
        k2 = evaluar_funcion_xy(func_str, xi + 0.5*h, yi + 0.5*h*k1)
        k3 = evaluar_funcion_xy(func_str, xi + 0.5*h, yi + 0.5*h*k2)
        k4 = evaluar_funcion_xy(func_str, xi + h, yi + h*k3)
        
        # Promedio ponderado
        pendiente_prom = (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        yi_new = yi + h * pendiente_prom
        xi_new = xi + h
        
        # Exacta y Error
        exacta_val = None
        error_val = None
        if sol_exacta_str:
            try:
                exacta_val = evaluar_funcion(sol_exacta_str, xi_new)
                error_val = abs(yi_new - exacta_val)
            except: pass
        
        detalle = {
            "iter": i+1,
            "x": xi,
            "y": yi,
            "k1": k1, "k2": k2, "k3": k3, "k4": k4,
            "y_new": yi_new,
            "exacta": exacta_val,
            "error": error_val
        }
        log_pasos.append(detalle)
        
        xi = xi_new
        yi = yi_new
        
        x_vals.append(xi)
        y_vals.append(yi)

    return np.array(x_vals), np.array(y_vals), log_pasos


def verificar_orden_edo(func_str, x0, y0, h, n, metodo, sol_exacta_str):
    """
    Compara error global final con paso h y paso h/2.
    """
    if not sol_exacta_str:
        return "Se requiere la Solución Exacta para verificar el orden de convergencia."

    # Ejecución 1: Paso h
    if metodo == "Euler":
        vals_x1, vals_y1, _ = metodo_euler(func_str, x0, y0, h, n, sol_exacta_str)
    else:
        vals_x1, vals_y1, _ = metodo_rk4(func_str, x0, y0, h, n, sol_exacta_str)
    
    y_final_1 = vals_y1[-1]
    x_final = vals_x1[-1]
    
    y_exacta = evaluar_funcion(sol_exacta_str, x_final)
    error_1 = abs(y_final_1 - y_exacta)

    # Ejecución 2: Paso h/2 (doble de iteraciones)
    h2 = h / 2
    n2 = n * 2
    
    if metodo == "Euler":
        vals_x2, vals_y2, _ = metodo_euler(func_str, x0, y0, h2, n2, sol_exacta_str)
    else:
        vals_x2, vals_y2, _ = metodo_rk4(func_str, x0, y0, h2, n2, sol_exacta_str)
        
    y_final_2 = vals_y2[-1]
    error_2 = abs(y_final_2 - y_exacta)
    
    reporte = f"--- Verificación de Orden ({metodo}) ---\n"
    reporte += f"x final evaluado: {x_final:.6f}\n"
    reporte += f"Valor Exacto: {y_exacta:.8f}\n\n"
    
    reporte += f"Con paso h = {h}:\n"
    reporte += f"  y_aprox = {y_final_1:.8f}\n"
    reporte += f"  Error(h) = {error_1:.8e}\n\n"
    
    reporte += f"Con paso h/2 = {h2}:\n"
    reporte += f"  y_aprox = {y_final_2:.8f}\n"
    reporte += f"  Error(h/2) = {error_2:.8e}\n\n"
    
    if error_2 > 0:
        ratio = error_1 / error_2
        # E(h) approx C * h^p
        # Ratio = (h^p) / ((h/2)^p) = 2^p
        # p = log2(Ratio)
        p = np.log2(ratio)
        reporte += f"Ratio de error (E1/E2): {ratio:.4f}\n"
        reporte += f"Orden estimado p ≈ {p:.4f}\n"
        
        esperado = 1 if metodo == "Euler" else 4
        reporte += f"Orden Teórico Esperado: {esperado}\n"
    else:
        reporte += "Error es 0. No se puede estimar orden (solución exacta alcanzada).\n"
        
    return reporte
