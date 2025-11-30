import numpy as np

# ======================================================
# UTILIDADES
# ======================================================


def evaluar_funcion(funcion_str, x_val):
    """Evalúa una función string de forma segura usando Numpy."""
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
