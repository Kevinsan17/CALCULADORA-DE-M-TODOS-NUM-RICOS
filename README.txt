========================================================================
             CALCULADORA DE MÉTODOS NUMÉRICOS (Python + GUI)
========================================================================

AUTOR: Kevin Japón & Matías Alcocer
ASIGNATURA: Métodos Numéricos
VERSION: 2.0 (Completa - Primer y Segundo Bimestre)
FECHA: Febrero 2026

------------------------------------------------------------------------
1. DESCRIPCIÓN GENERAL
------------------------------------------------------------------------
Esta aplicación es una calculadora gráfica avanzada diseñada para la asignatura
de Métodos Numéricos. Implementa una amplia gama de algoritmos matemáticos
para resolver problemas de álgebra lineal, ecuaciones no lineales, interpolación,
integración, derivación y ecuaciones diferenciales ordinarias (EDO).

La interfaz gráfica moderna (construida con CustomTkinter) permite visualizar
los resultados paso a paso, gráficas interactivas y análisis de errores.

------------------------------------------------------------------------
2. ESTRUCTURA DEL PROYECTO
------------------------------------------------------------------------
El proyecto se organiza en dos archivos principales:

- main.py:     Contiene la lógica de la Interfaz Gráfica de Usuario (GUI).
               Gestiona las pestañas, botones, gráficas y entradas de datos.
               
- calculos.py: Librería de funciones matemáticas. Implementa todos los
               algoritmos numéricos utilizando NumPy para alta precisión.

------------------------------------------------------------------------
3. REQUISITOS E INSTALACIÓN
------------------------------------------------------------------------
Para ejecutar este programa, necesita Python 3.8 o superior.
Las librerías requeridas son:

- customtkinter  (Interfaz)
- numpy          (Cálculos matemáticos)
- matplotlib     (Gráficas)
- packaging      (Dependencia común)

Instalación rápida:
   pip install customtkinter numpy matplotlib

Ejecucción:
   python main.py

------------------------------------------------------------------------
4. GUÍA DE USO - PRINCIPALES FUNCIONALIDADES
------------------------------------------------------------------------
La calculadora está dividida en dos grandes pestañas: "Primer Parcial" y 
"Segundo Parcial".

========================================================================
   A) PRIMER PARCIAL (Álgebra Lineal y Ecuaciones No Lineales)
========================================================================

1. Sistemas de Ecuaciones Lineales:
   - Gauss Simple, Gauss-Seidel, Pivoteo Parcial.
   - Formato Matriz A: Filas separadas por punto y coma (;), columnas por coma (,).
     Ejemplo: 3,2; 1,4   (Matriz 2x2)
   - Formato Vector b: Números separados por comas.

2. Ecuaciones No Lineales (Raíces):
   - Métodos: Bisección, Newton-Raphson, Secante, Secante Modificada.
   - Entrada de Función: Sintaxis Python (ver sección de Sintaxis).
   - Salida: Raíz aproximada y número de iteraciones.

3. Regresiones (Ajuste de Curvas):
   - Lineal y Polinomial (Grado n).
   - Entrada: Listas de datos X e Y separados por comas.
   - Salida: Ecuación del modelo, Coeficiente R^2 y Gráfica.

========================================================================
   B) SEGUNDO PARCIAL (Cálculo Numérico Avanzado)
========================================================================

1. INTERPOLACIÓN
   ---------------------------------------------------------------------
   Permite estimar valores intermedios entre puntos conocidos.
   - Métodos: 
     * Lagrange: Muestra la construcción del polinomio paso a paso.
     * Newton (Diferencias Divididas): Muestra el polinomio final.
   - Validación: Permite ingresar puntos de control extra para calcular 
     errores de interpolación.

2. INTEGRACIÓN NUMÉRICA
   ---------------------------------------------------------------------
   Calcula la integral definida de una función o datos tabulados.
   - Métodos: Trapecio (Simple/Múltiple), Simpson 1/3, Simpson 3/8.
   - Modos de Entrada:
     * Función f(x): Permite definir límites (a, b) y número de pasos (n) o tamaño (h).
     * Datos Tabulados: Detecta automáticamente el paso h (si es uniforme).
   - Análisis: 
     * Convergencia: Compara resultados con n y 2n para estimar el orden.
     * Cotas de Error: Estima el error máximo usando derivadas numéricas.

3. DERIVACIÓN NUMÉRICA
   ---------------------------------------------------------------------
   Calcula la derivada f'(x) en un punto dado.
   - Métodos Regulares (Paso h constante):
     * Diferencia Adelante (Forward)
     * Diferencia Atrás (Backward)
     * Diferencia Centrada (Centered)
     * Selección de Orden de Error: O(h), O(h^2), O(h^4) según el método.
   - Datos Irregulares:
     * Irregular (3 puntos): Usa interpolación de Lagrange para derivar con 
       datos no equiespaciados.
   - Herramientas Extra:
     * Análisis de Paso (h): Muestra cómo cambia el error al variar h, útil
       para detectar errores de redondeo vs truncamiento.

4. ECUACIONES DIFERENCIALES ORDINARIAS (EDO)
   ---------------------------------------------------------------------
   Resuelve problemas de valor inicial: y' = f(x,y), y(x0) = y0.
   - Métodos: Euler (Primer orden) y Runge-Kutta 4 (Cuarto orden).
   - Condición de Parada: Por valor final de x o número de iteraciones.
   - Validación Exacta:
     * Puede ingresar la Solución Analítica y(x) (opcional).
     * El programa calculará el Error Absoluto en cada paso.
     * Gráfica comparativa: Numérica (punteada azul) vs Exacta (verde).
   - Verificación de Orden: Botón especial que ejecuta el método con paso h
     y paso h/2 para confirmar empíricamente el orden de convergencia.

------------------------------------------------------------------------
5. SINTAXIS MATEMÁTICA SOPORTADA
------------------------------------------------------------------------
Todas las entradas de funciones utilizan la sintaxis estándar de Python/NumPy.
Asegúrese de escribir las operaciones explícitamente.

Variables:
 - Utilice 'x' para funciones de una variable f(x).
 - Utilice 'x' e 'y' para EDOs f(x,y).

Operadores:
 - Suma: +
 - Resta: -
 - Multiplicación: *  (Ejemplo: 2*x, NO 2x)
 - División: /
 - Potencia: ** (Ejemplo: x**2 para x al cuadrado)

Funciones Disponibles:
 - Trigonométricas: sin(x), cos(x), tan(x)
 - Exponenciales: exp(x), log(x)  (log es logaritmo natural)
 - Raíz: sqrt(x)
 - Constantes: pi, e

Ejemplos Válidos:
 - x**2 + 2*x + 1
 - exp(-x) * sin(2*x)
 - (x - y) / 2
 - sqrt(x**2 + 1)

------------------------------------------------------------------------
6. SOLUCIÓN DE PROBLEMAS COMUNES
------------------------------------------------------------------------
- "name 'np' is not defined": Asegúrese de no borrar "import numpy as np"
  en los archivos fuente.
- "ValueError: h must be > 0": Revise que el límite superior 'b' sea mayor
  que 'a' y que n sea positivo.
- Gráficas no aparecen: Asegúrese de tener matplotlib instalado.
- Errores de sintaxis en función: Revise que multiplicaciones como 2x sean 2*x.

========================================================================
