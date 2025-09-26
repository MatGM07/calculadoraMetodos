import sympy as sp

def diferenciacionNumericaPorTabla(X, Y, Xi, h, ordenDerivada=1):
    """
    Método para calcular derivadas numéricas usando diferencias divididas.
    Soporta fórmulas hacia adelante, hacia atrás y centradas.

    Parámetros:
    X (list): Lista de valores x.
    Y (list): Lista de valores f(x).
    Xi (float): Valor de x en el que calcular la derivada.
    h (float): Paso entre valores x.
    ordenDerivada (int): Orden de la derivada (1, 2, 3, o 4).

    Retorna:
    float: Aproximación de la derivada en Xi.
    """
    i = X.index(Xi)

    if ordenDerivada == 1:  # Primera derivada
        if i >= 2 and i <= len(X) - 3:  # Centrada
            return (-Y[i+2] + 8*Y[i+1] - 8*Y[i-1] + Y[i-2]) / (12 * h)
        elif i < 2:  # Hacia adelante
            if i + 2 < len(X):
                return (-3*Y[i] + 4*Y[i+1] - Y[i+2]) / (2 * h)
            else:
                return (Y[i+1] - Y[i]) / h
        else:  # Hacia atrás
            if i - 2 >= 0:
                return (3*Y[i] - 4*Y[i-1] + Y[i-2]) / (2 * h)
            else:
                return (Y[i] - Y[i-1]) / h

    elif ordenDerivada == 2:  # Segunda derivada
        if i >= 2 and i <= len(X) - 3:  # Centrada
            return (-Y[i+2] + 16*Y[i+1] - 30*Y[i] + 16*Y[i-1] - Y[i-2]) / (12 * h**2)
        elif i < 2:  # Hacia adelante
            if i + 2 < len(X):
                return (Y[i] - 2*Y[i+1] + Y[i+2]) / h**2
            else:
                raise ValueError("No hay suficientes datos para diferencias hacia adelante.")
        else:  # Hacia atrás
            if i - 2 >= 0:
                return (Y[i-2] - 2*Y[i-1] + Y[i]) / h**2
            else:
                raise ValueError("No hay suficientes datos para diferencias hacia atrás.")

    elif ordenDerivada == 3:  # Tercera derivada
        if i >= 3 and i <= len(X) - 4:  # Centrada
            return (-Y[i+3] + 8*Y[i+2] - 13*Y[i+1] + 13*Y[i-1] - 8*Y[i-2] + Y[i-3]) / (8 * h**3)
        elif i < 3:  # Hacia adelante
            if i + 3 < len(X):
                return (-Y[i+3] + 3*Y[i+2] - 3*Y[i+1] + Y[i]) / h**3
            else:
                raise ValueError("No hay suficientes datos para diferencias hacia adelante.")
        else:  # Hacia atrás
            if i - 3 >= 0:
                return (Y[i-3] - 3*Y[i-2] + 3*Y[i-1] - Y[i]) / h**3
            else:
                raise ValueError("No hay suficientes datos para diferencias hacia atrás.")

    elif ordenDerivada == 4:  # Cuarta derivada
        if i >= 3 and i <= len(X) - 4:  # Centrada
            return (-Y[i+3] + 12*Y[i+2] - 39*Y[i+1] + 56*Y[i] - 39*Y[i-1] + 12*Y[i-2] - Y[i-3]) / (6 * h**4)
        elif i < 3:  # Hacia adelante
            if i + 4 < len(X):
                return (Y[i] - 4*Y[i+1] + 6*Y[i+2] - 4*Y[i+3] + Y[i+4]) / h**4
            else:
                raise ValueError("No hay suficientes datos para diferencias hacia adelante.")
        else:  # Hacia atrás
            if i - 4 >= 0:
                return (Y[i-4] - 4*Y[i-3] + 6*Y[i-2] - 4*Y[i-1] + Y[i]) / h**4
            else:
                raise ValueError("No hay suficientes datos para diferencias hacia atrás.")

    else:
        raise ValueError("Orden de derivada no soportado. Solo se permite 1, 2, 3 o 4.")


def diferenciacionNumericaPorFuncion(funcion, Xi, h, ordenDerivada=1):
    X=[]
    Y=[]
    for i in range(-3,4):
        X.append(Xi-i*h)
        Y.append(funcion(Xi-i*h))

    return diferenciacionNumericaPorTabla(X, Y, Xi, h, ordenDerivada)

