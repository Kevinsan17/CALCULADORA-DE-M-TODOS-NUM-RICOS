========================================================================
             CALCULADORA DE MÉTODOS NUMÉRICOS (Python + GUI)
========================================================================

AUTORES: Kevin Japón y Matías Alcocer
ASIGNATURA: Métodos Numéricos
FECHA: Noviembre 2025

------------------------------------------------------------------------
1. DESCRIPCIÓN
------------------------------------------------------------------------
Esta aplicación es una calculadora gráfica que implementa diversos algoritmos 
numéricos para la resolución de:
1. Sistemas de Ecuaciones Lineales (Gauss Simple, Gauss-Seidel, Pivoteo).
2. Ecuaciones No Lineales (Bisección, Newton-Raphson, Secante).
3. Ajuste de Curvas (Regresión Lineal y Polinomial).

El proyecto está estructurado en dos archivos para separar la lógica de la interfaz:
- main.py: Contiene la interfaz gráfica (CustomTkinter).
- calculos.py: Contiene los algoritmos matemáticos (NumPy).

------------------------------------------------------------------------
2. REQUISITOS DEL SISTEMA
------------------------------------------------------------------------
Para ejecutar este programa, necesita tener instalado Python (3.8 o superior).
Las librerías externas necesarias son:

- customtkinter
- numpy
- matplotlib
- packaging (a veces requerido por matplotlib)

Puede instalarlas ejecutando el siguiente comando en su terminal:

   pip install customtkinter numpy matplotlib

------------------------------------------------------------------------
3. CÓMO EJECUTAR
------------------------------------------------------------------------
1. Asegúrese de que los archivos 'main.py' y 'calculos.py' estén en la misma carpeta.
2. Abra una terminal en dicha carpeta.
3. Ejecute el archivo principal:

   python main.py

------------------------------------------------------------------------
4. GUÍA DE USO Y FORMATO DE DATOS
------------------------------------------------------------------------
A continuación se detalla cómo ingresar los datos en los campos de texto:

A) MATRICES Y VECTORES (Sistemas de Ecuaciones):
   - Matriz A: Separe las columnas con comas (,) y las filas con punto y coma (;).
     Ejemplo (2x2):  3,1; 1,4
     Esto representa:
     | 3  1 |
     | 1  4 |
   
   - Vectores (b, x0): Separe los números con comas.
     Ejemplo: 5,6

   *Nota para Gauss-Seidel: El programa verifica automáticamente si la matriz
    es diagonalmente dominante y mostrará una advertencia si no lo es.

B) FUNCIONES (Ecuaciones No Lineales):
   - Utilice sintaxis de Python. Las funciones matemáticas disponibles son:
     sin, cos, tan, exp (exponencial), log (logaritmo natural), sqrt, pi, e.
     La variable siempre debe ser 'x'.
     
     Ejemplo 1: x**2 - 4
     Ejemplo 2: exp(-x) - x
     Ejemplo 3: sin(x) + cos(x)

C) DATOS PARA REGRESIÓN:
   - Ingrese los datos separados por comas.
   - Datos X: 1, 2, 3, 4
   - Datos Y: 2.1, 4.2, 6.1, 8.0
