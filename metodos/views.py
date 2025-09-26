from django.http import HttpResponse
from django.shortcuts import render
import sympy as sim
from sympy import sympify, symbols, pi, E, N
from sympy.parsing.latex import parse_latex
import numpy as np
import plotly.graph_objs as go
from tabulate import tabulate as tab
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application
import re

def normalizar_latex(expr):
    """
    Convierte una expresión LaTeX a texto plano compatible con SymPy.
    """
    expr = expr.replace('\\pi', 'pi')  # Reemplaza pi
    expr = re.sub(r'e\^\{([^{}]+)\}', r'exp(\1)', expr)
    expr = expr.replace('\\ln', 'log')  # Reemplaza ln con log
    expr = expr.replace('\\log', 'log')  # Reemplaza log explícito
    expr = expr.replace('\\cos', 'cos')  # Reemplaza log explícito
    expr = expr.replace('\\sin', 'sin')  # Reemplaza log explícito
    expr = expr.replace('\\tan', 'tan')
    expr = expr.replace('{', '(')# Reemplaza log explícito
    expr = expr.replace('}', ')')
    expr = expr.replace('^', '**')    # Reemplaza potencias
    expr = expr.replace(' ', '')      # Elimina espacios
    return expr
def secante(request):
    return render(request, 'secante.html')

def index(request):
    return render(request, 'index.html')

def falsaPosicion(request):
    return render(request, 'falsaPosicion.html')

def falsaPosicionResuelta(request):
    if request.method == 'POST':
        latex_expr = request.POST.get('expresion-latex')
        print(latex_expr)

        try:
            latex_expr = normalizar_latex(latex_expr)
            print(f"Expresión LaTeX normalizada: {latex_expr}")
            transformations = (standard_transformations + (implicit_multiplication_application,))
            # Parseo de la expresión normalizada
            sympy_expr = parse_expr(latex_expr, transformations=transformations)
            print(f"Expresión SymPy: {sympy_expr}")
            x = symbols('x')

            expresionxl = request.POST.get('xl')
            exprxl = sim.sympify(expresionxl)
            xl = exprxl.evalf()

            expresionxu = request.POST.get('xu')
            exprxu = sim.sympify(expresionxu)
            xu = exprxu.evalf()

            expresionE = request.POST.get('error')
            exprE = sim.sympify(expresionE)
            tol = exprE.evalf()

            lista_error = []
            lista_xi = []
            lista_fxi = []

            Xr = xl

            for i in range(5000):
                Xr_old = Xr
                f_Xl = sympy_expr.evalf(subs={x: xl})
                f_Xu = sympy_expr.evalf(subs={x: xu})

                # Cálculo de Xr con falsa posición
                Xr = xu - (f_Xu * (xl - xu)) / (f_Xl - f_Xu)
                f_Xr = sympy_expr.evalf(subs={x: Xr})

                # Calcular error
                if i > 0:
                    print("valor de i: ")
                    print(i)
                    error = abs((Xr - Xr_old) / Xr) * 100
                else:
                    error = 100


                # Verificar si se cumple la tolerancia
                if f_Xr == 0 or (error < tol):
                    print(tol)
                    print(error)
                    print(f_Xr)
                    break

                lista_xi.append(float(Xr))
                lista_fxi.append(float(f_Xr))
                lista_error.append(float(error))

                # Siguiente intervalo
                if f_Xl * f_Xr < 0:
                    xu = Xr
                else:
                    xl = Xr

            valores_x = np.linspace(-10, 10, 500)
            resultados_y = [float(sympy_expr.evalf(subs={x: val})) for val in valores_x]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=valores_x, y=resultados_y, mode='lines', name='Función'))

            for xi_intermedio in lista_xi[:-1]:
                yi_intermedio = float(sympy_expr.evalf(subs={x: xi_intermedio}))
                fig.add_trace(go.Scatter(x=[xi_intermedio], y=[yi_intermedio], mode='markers',
                                         marker=dict(color='green'), name='Punto Intermedio'))

            xi_final = lista_xi[-1]
            yi_final = float(sympy_expr.evalf(subs={x: xi_final}))
            fig.add_trace(go.Scatter(x=[xi_final], y=[yi_final], mode='markers',
                                     marker=dict(color='red'), name='Punto Óptimo'))

            fig.update_layout(
                title='Obtenciónd de raiz usando el Metodo de la falsa posición',
                xaxis_title='Valor de x',
                yaxis_title='Valor de la función',
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='red'),
                yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='blue'),
                width=1000,
                height=800
            )

            plot_div = fig.to_html(full_html=False)

            return render(request, 'falsaPosicionResuelta.html', {
                'plot_div': plot_div,
                'lista_datos': zip(lista_xi, lista_fxi, lista_error),
                'valor_final_x': lista_xi[-1]
                # Combina las listas
            })
        except ZeroDivisionError as zde:
            return render(request, 'falsaPosicion.html', {
                'fatalerror': "Error: División por cero durante la iteración. Intenta con valores iniciales diferentes."
            })


        except (ValueError, TypeError) as e:

            return render(request, 'falsaPosicion.html', {

                'fatalerror': f"No se ha encontrado raíz real en los valores iniciales suministrados. Error: {str(e)}"

            })
    return render(request, 'falsaPosicion.html', {
        'fatalerror': "Error Desconocido"
    })

def integracion(request):
    return render(request, 'integracion.html')

def integracionResuelta(request):
    if request.method == 'POST':
        latex_expr = request.POST.get('expresion-latex')

        expresionA = request.POST.get('a')
        exprA = sim.sympify(expresionA)
        a = exprA.evalf()  # Evaluación numérica


        expresionB = request.POST.get('b')
        exprB = sim.sympify(expresionB)
        b = exprB.evalf()  # Evaluación numérica


        k = int(request.POST.get('k'))

        try:
            latex_expr = normalizar_latex(latex_expr)
            print(f"Expresión LaTeX normalizada: {latex_expr}")
            transformations = (standard_transformations + (implicit_multiplication_application,))
            # Parseo de la expresión normalizada
            sympy_expr = parse_expr(latex_expr, transformations=transformations)
            print(f"Expresión SymPy: {sympy_expr}")
            x = symbols('x')

            def integralReglaDelTrapecio(sympy_expr, n, a, b):

                deltaX = (b - a) / n
                xi = []
                fXi = []
                Area = 0
                for i in range(n + 1):
                    xi.append(a + deltaX * i)
                for i in range(len(xi)):
                    sympy_exprAux = sympy_expr.subs(x, xi[i])
                    fXi.append(N(sympy_exprAux))
                for i in range(n):
                    sympy_exprAux1 = sympy_expr.subs(x, xi[i])

                    sympy_exprAux2 = sympy_expr.subs(x, xi[i+1])


                    Area += deltaX * (N(sympy_exprAux1) + N(sympy_exprAux2)) / 2

                return Area

            def formulaRomberg(k, aprox1, aprox2):
                return (4 ** (k - 1) * aprox1 - aprox2) / (4 ** (k - 1) - 1)

            def integralMetodoRomberg(sympy_expr, K, a, b):

                aproximacionesTrapecio = []
                for i in range(K):
                    aproximacionesTrapecio.append(integralReglaDelTrapecio(sympy_expr, 2 ** i, a, b))

                matriz = [[0 for _ in range(K)] for _ in range(K)]
                for i in range(K):
                    matriz[i][0] = aproximacionesTrapecio[i]

                for k in range(K):
                    if k == 0:
                        continue
                    for j in range(K - k):
                        matriz[j][k] = formulaRomberg(k + 1, matriz[j + 1][k - 1], matriz[j][k - 1])


                return matriz[0][K - 1], matriz

            resultado, matriz_romberg = integralMetodoRomberg(sympy_expr, k, a, b)

            return render(request, 'integracionResuelta.html', { 'valor_final_x': resultado , 'matriz_romberg': matriz_romberg})


        except ZeroDivisionError as zde:
            return render(request, 'secante.html', {
                'fatalerror': "Error: División por cero durante la iteración. Intenta con valores iniciales diferentes."

            })

        except (ValueError, TypeError):
            return render(request, 'secante.html', {
                'fatalerror': "Error desconocido. Verificar datos"

            })

    return render(request, 'integracionResuelta.html',{
                'fatalerror': "Error Desconocido"
            })
def diferenciacion(request):
    return render(request, 'diferenciacion.html')

def diferenciacionTabla(request):


    return render(request, 'diferenciacionTabla.html')

def diferenciacionTablaResuelta(request):
    if request.method == 'POST':
        try:

            X = list(map(float, request.POST.get('valores_x').split(',')))
            Y = list(map(float, request.POST.get('valores_y').split(',')))
            irregular =  request.POST.get('tabla_irregular_valor')


            expresionP = request.POST.get('p')
            exprP = sim.sympify(expresionP)
            punto = exprP.evalf()  # Evaluación numérica

            print(irregular)

            if (irregular == "1"):

                print("boyacoman")
                term1 = Y[0] * (2 * punto - X[1] - X[2]) / ((X[0] - X[1]) * (X[0] - X[2]))
                term2 = Y[1] * (2 * punto - X[0] - X[2]) / ((X[1] - X[0]) * (X[1] - X[2]))
                term3 = Y[2] * (2 * punto - X[0] - X[1]) / ((X[2] - X[0]) * (X[2] - X[1]))

                return render(request, 'diferenciacionTablaResuelta.html', {'resultado': term1 + term2 + term3})

            else:
                ordenDerivada = int(request.POST.get('orden'))
                h = float(request.POST.get('h'))
                i = X.index(punto)

                respuesta=667
                if ordenDerivada == 1:  # Primera derivada
                    if i >= 2 and i <= len(X) - 3:  # Centrada
                        respuesta = (-Y[i + 2] + 8 * Y[i + 1] - 8 * Y[i - 1] + Y[i - 2]) / (12 * h)
                        print(respuesta)
                    elif i < 2:  # Hacia adelante
                        if i + 2 < len(X):
                            respuesta = (-3 * Y[i] + 4 * Y[i + 1] - Y[i + 2]) / (2 * h)
                        else:
                            respuesta = (Y[i + 1] - Y[i]) / h
                    else:  # Hacia atrás
                        if i - 2 >= 0:
                            respuesta = (3 * Y[i] - 4 * Y[i - 1] + Y[i - 2]) / (2 * h)
                        else:
                            respuesta = (Y[i] - Y[i - 1]) / h

                elif ordenDerivada == 2:  # Segunda derivada
                    if i >= 2 and i <= len(X) - 3:  # Centrada
                        respuesta = (-Y[i + 2] + 16 * Y[i + 1] - 30 * Y[i] + 16 * Y[i - 1] - Y[i - 2]) / (12 * h ** 2)
                    elif i < 2:  # Hacia adelante
                        if i + 2 < len(X):
                            respuesta = (Y[i] - 2 * Y[i + 1] + Y[i + 2]) / h ** 2
                    else:  # Hacia atrás
                        if i - 2 >= 0:
                            respuesta = (Y[i - 2] - 2 * Y[i - 1] + Y[i]) / h ** 2

                elif ordenDerivada == 3:  # Tercera derivada
                    print("jaijaj3")
                    print(i)
                    if i >= 3 and i <= len(X) - 4:  # Centrada
                        respuesta = (-Y[i + 3] + 8 * Y[i + 2] - 13 * Y[i + 1] + 13 * Y[i - 1] - 8 * Y[i - 2] + Y[
                            i - 3]) / (
                                            8 * h ** 3)
                    elif i < 3:  # Hacia adelante
                        if i + 3 < len(X):
                            print("jaijaj")
                            respuesta = (-Y[i + 3] + 3 * Y[i + 2] - 3 * Y[i + 1] + Y[i]) / h ** 3
                    else:  # Hacia atrás
                        if i - 3 >= 0:
                            respuesta = (Y[i - 3] - 3 * Y[i - 2] + 3 * Y[i - 1] - Y[i]) / h ** 3
                elif ordenDerivada == 4:  # Cuarta derivada
                    if i >= 3 and i <= len(X) - 4:  # Centrada
                        respuesta = (-Y[i + 3] + 12 * Y[i + 2] - 39 * Y[i + 1] + 56 * Y[i] - 39 * Y[i - 1] + 12 * Y[
                            i - 2] - Y[
                                         i - 3]) / (6 * h ** 4)
                    elif i < 3:  # Hacia adelante
                        if i + 4 < len(X):
                            respuesta = (Y[i] - 4 * Y[i + 1] + 6 * Y[i + 2] - 4 * Y[i + 3] + Y[i + 4]) / h ** 4
                    else:  # Hacia atrás
                        if i - 4 >= 0:
                            respuesta = (Y[i - 4] - 4 * Y[i - 3] + 6 * Y[i - 2] - 4 * Y[i - 1] + Y[i]) / h ** 4

                else:
                    return render(request, 'diferenciacionTabla.html', {
                        'fatalerror': "Error: Orden no soportado"
                    })

                return render(request, 'diferenciacionTablaResuelta.html', {'resultado': respuesta})

        except ZeroDivisionError as zde:
            return render(request, 'diferenciacionTabla.html', {
                'fatalerror': "Error: División por cero durante la iteración. Intenta con valores iniciales diferentes."
            })


        except (ValueError, TypeError) as e:

            return render(request, 'diferenciacionTabla.html', {

                'fatalerror': f"No se ha encontrado raíz real en los valores iniciales suministrados. Error: {str(e)}"

            })

    return render(request, 'diferenciacionTabla.html', {
        'fatalerror': "Error Desconocido"
    })

def diferenciacionResuelta(request):
    if request.method == 'POST':
        try:

            latex_expr = request.POST.get('expresion-latex')
            latex_expr = normalizar_latex(latex_expr)
            print(f"Expresión LaTeX normalizada: {latex_expr}")
            transformations = (standard_transformations + (implicit_multiplication_application,))
            # Parseo de la expresión normalizada
            sympy_expr = parse_expr(latex_expr, transformations=transformations)
            x = symbols('x')

            ordenDerivada = int(request.POST.get('orden'))
            h = float(request.POST.get('h'))

            expresionP = request.POST.get('p')
            exprP = sim.sympify(expresionP)
            punto = float(exprP.evalf())  # Evaluación numérica
            print("jiajaja2")
            X = []
            Y = []
            for i in range(-3, 4):
                X.append(punto + i * h)

                newY = float(punto + float(i) * h)

                sympy_exprAux = sympy_expr.subs(x, newY)

                Y.append(sympy_exprAux)


            i = X.index(punto)
            print(X)
            print(Y)
            respuesta = 0

            if ordenDerivada == 1:  # Primera derivada
                if i >= 2 and i <= len(X) - 3:  # Centrada
                    respuesta = (-Y[i + 2] + 8 * Y[i + 1] - 8 * Y[i - 1] + Y[i - 2]) / (12 * h)
                    print(respuesta)
                elif i < 2:  # Hacia adelante
                    if i + 2 < len(X):
                        respuesta = (-3 * Y[i] + 4 * Y[i + 1] - Y[i + 2]) / (2 * h)
                    else:
                        respuesta = (Y[i + 1] - Y[i]) / h
                else:  # Hacia atrás
                    if i - 2 >= 0:
                        respuesta = (3 * Y[i] - 4 * Y[i - 1] + Y[i - 2]) / (2 * h)
                    else:
                        respuesta = (Y[i] - Y[i - 1]) / h

            elif ordenDerivada == 2:  # Segunda derivada
                if i >= 2 and i <= len(X) - 3:  # Centrada
                    respuesta = (-Y[i + 2] + 16 * Y[i + 1] - 30 * Y[i] + 16 * Y[i - 1] - Y[i - 2]) / (12 * h ** 2)
                elif i < 2:  # Hacia adelante
                    if i + 2 < len(X):
                        respuesta = (Y[i] - 2 * Y[i + 1] + Y[i + 2]) / h ** 2
                else:  # Hacia atrás
                    if i - 2 >= 0:
                        respuesta = (Y[i - 2] - 2 * Y[i - 1] + Y[i]) / h ** 2

            elif ordenDerivada == 3:  # Tercera derivada
                if i >= 3 and i <= len(X) - 4:  # Centrada
                    respuesta = (-Y[i + 3] + 8 * Y[i + 2] - 13 * Y[i + 1] + 13 * Y[i - 1] - 8 * Y[i - 2] + Y[i - 3]) / (
                                8 * h ** 3)
                elif i < 3:  # Hacia adelante
                    if i + 3 < len(X):
                        respuesta = (-Y[i + 3] + 3 * Y[i + 2] - 3 * Y[i + 1] + Y[i]) / h ** 3
                else:  # Hacia atrás
                    if i - 3 >= 0:
                        respuesta = (Y[i - 3] - 3 * Y[i - 2] + 3 * Y[i - 1] - Y[i]) / h ** 3
            elif ordenDerivada == 4:  # Cuarta derivada
                if i >= 3 and i <= len(X) - 4:  # Centrada
                    respuesta = (-Y[i + 3] + 12 * Y[i + 2] - 39 * Y[i + 1] + 56 * Y[i] - 39 * Y[i - 1] + 12 * Y[i - 2] - Y[
                        i - 3]) / (6 * h ** 4)
                elif i < 3:  # Hacia adelante
                    if i + 4 < len(X):
                        respuesta = (Y[i] - 4 * Y[i + 1] + 6 * Y[i + 2] - 4 * Y[i + 3] + Y[i + 4]) / h ** 4
                else:  # Hacia atrás
                    if i - 4 >= 0:
                        respuesta = (Y[i - 4] - 4 * Y[i - 3] + 6 * Y[i - 2] - 4 * Y[i - 1] + Y[i]) / h ** 4

            else:
                return render(request, 'diferenciacion.html', {
                    'fatalerror': "Error: Orden no soportado"
                })

            return render(request, 'diferenciacionResuelta.html', {'resultado': respuesta})

        except ZeroDivisionError as zde:
            return render(request, 'diferenciacion.html', {
                'fatalerror': "Error: División por cero durante la iteración. Intenta con valores iniciales diferentes."
            })


        except (ValueError, TypeError) as e:

            return render(request, 'diferenciacion.html', {

                'fatalerror': f"No se ha encontrado raíz real en los valores iniciales suministrados. Error: {str(e)}"

            })

def optimizacion(request):
    return render(request, 'optimizacion.html')

def optimizacionResuelta(request):
    if request.method == 'POST':
        latex_expr = request.POST.get('expresion-latex')
        print(latex_expr)

        try:
            latex_expr = normalizar_latex(latex_expr)
            print(f"Expresión LaTeX normalizada: {latex_expr}")
            transformations = (standard_transformations + (implicit_multiplication_application,))
            # Parseo de la expresión normalizada
            sympy_expr = parse_expr(latex_expr, transformations=transformations)
            print(f"Expresión SymPy: {sympy_expr}")
            x = symbols('x')

            primera_derivada = sim.diff(sympy_expr, x)


            segunda_derivada = sim.diff(primera_derivada, x)

            expresionxi = request.POST.get('x0')
            exprxi = sim.sympify(expresionxi)
            xi = exprxi.evalf()  # Evaluación numérica

            expresionE = request.POST.get('error')
            exprE = sim.sympify(expresionE)
            tolerancia = exprE.evalf()


            lista_error = []
            lista_xi = []
            lista_fxi = []
            lista_fxiderivada = []
            lista_fxisegundaderivada = []

            for i in range(5000):
                # CALCULO DE PRIMERA Y SEGUNDA DERIVADA
                valor_primera_derivada = float(primera_derivada.evalf(subs={x: xi}))
                valor_segunda_derivada = float(segunda_derivada.evalf(subs={x: xi}))
                if valor_segunda_derivada == 0:
                    print("ERROR: LA SEGUNDA DERIVADA ES CERO, NO SE PUEDE CONTINUAR CON EL MÉTODO.")
                    break

                # CALCULO DEL NUEVO VALOR XI
                xi_nuevo = xi - (valor_primera_derivada / valor_segunda_derivada)

                # CALCULO DEL ERROR
                error = abs((xi_nuevo - xi) / xi_nuevo) * 100 if xi_nuevo != 0 else 0

                fxi = float(sympy_expr.evalf(subs={x: xi}))

                lista_xi.append(float(xi))
                lista_fxi.append(float(fxi))
                lista_fxiderivada.append(float(valor_primera_derivada))
                lista_fxisegundaderivada.append(float(valor_segunda_derivada))
                lista_error.append(float(error))


                # ALMACENAR RESULTADOS PARA LA TABLA

                # CONDICION DE TOLERANCIA
                if error < tolerancia:
                    break

                # ACTUALIZAR EL VALOR DE XI
                xi = xi_nuevo

            valores_x = np.linspace(-10, 10, 500)
            resultados_y = [float(sympy_expr.evalf(subs={x: val})) for val in valores_x]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=valores_x, y=resultados_y, mode='lines', name='Función'))

            for xi_intermedio in lista_xi[:-1]:
                yi_intermedio = float(sympy_expr.evalf(subs={x: xi_intermedio}))
                fig.add_trace(go.Scatter(x=[xi_intermedio], y=[yi_intermedio], mode='markers',
                                         marker=dict(color='green'), name='Punto Intermedio'))

            xi_final = lista_xi[-1]
            yi_final = float(sympy_expr.evalf(subs={x: xi_final}))
            fig.add_trace(go.Scatter(x=[xi_final], y=[yi_final], mode='markers',
                                     marker=dict(color='red'), name='Punto Óptimo'))

            fig.update_layout(
                title='Optimización usando el Metodo de Newton',
                xaxis_title='Valor de x',
                yaxis_title='Valor de la función',
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='red'),
                yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='blue'),
                width=1000,
                height=800
            )

            plot_div = fig.to_html(full_html=False)

            return render(request, 'optimizacionResuelta.html', {
                'plot_div': plot_div,
                'lista_datos': zip(lista_xi, lista_fxi, lista_fxiderivada, lista_fxisegundaderivada, lista_error),
                'valor_final_x': lista_xi[-1]
                # Combina las listas
            })
        except ZeroDivisionError as zde:
            return render(request, 'optimizacion.html', {
                'fatalerror': "Error: División por cero durante la iteración. Intenta con valores iniciales diferentes."
            })


        except (ValueError, TypeError) as e:

            return render(request, 'optimizacion.html', {

                'fatalerror': "Error Desconocido"

            })
    return render(request, 'optimizacion.html', {
        'fatalerror': "Error Desconocido"
    })

def ajuste(request):
    return render(request, 'ajuste.html')

def ajusteResuelta(request):
    if request.method == 'POST':
        try:

            X = list(map(float, request.POST.get('valores_x').split(',')))
            Y = list(map(float, request.POST.get('valores_y').split(',')))

            n = len(X)
            if len(Y) != n:
                raise ValueError("X y Y deben tener el mismo tamaño.")

            # Tabla de diferencias divididas
            diferencias_divididas = [list(Y)]  # Primera columna son los valores de Y
            for j in range(1, n):  # Iterar columnas
                columna = []
                for i in range(n - j):  # Calcular diferencias divididas
                    valor = (diferencias_divididas[j - 1][i + 1] - diferencias_divididas[j - 1][i]) / (X[i + j] - X[i])
                    columna.append(valor)
                diferencias_divididas.append(columna)

            # Construcción del polinomio
            x = sim.Symbol('x')  # Variable simbólica
            polinomio = diferencias_divididas[0][0]  # Primer término del polinomio
            for j in range(1, n):
                termino = diferencias_divididas[j][0]  # Coeficiente de la columna actual
                for i in range(j):  # Producto de (x - X[k]) para k = 0, ..., j-1
                    termino *= (x - X[i])
                polinomio += termino

            # Simplificar el polinomio
            polinomio_simplificado = sim.simplify(polinomio)

            # Graficar el polinomio simplificado
            valores_x = np.linspace(min(X) - 1, max(X) + 1, 5000)
            valores_y = [float(polinomio_simplificado.subs(x, val)) for val in valores_x]

            # Crear la gráfica
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=valores_x, y=valores_y, mode='lines', name='Polinomio de Newton'))

            # Añadir puntos originales
            fig.add_trace(
                go.Scatter(x=X, y=Y, mode='markers', marker=dict(color='red', size=10), name='Puntos Originales'))

            # Personalización del gráfico
            fig.update_layout(
                title='Interpolación Polinómica de Newton',
                xaxis_title='x',
                yaxis_title='f(x)',
                xaxis=dict(linecolor='black', linewidth=2),
                yaxis=dict(linecolor='black', linewidth=2),
                width=1000,
                height=800,
            )

            # Convertir gráfico a HTML
            plot_div = fig.to_html(full_html=False)

            polinomio_latex = sim.latex(polinomio_simplificado)

            return render(request, 'ajusteResuelta.html', {
                'plot_div': plot_div,
                'polinomio': polinomio_latex,
                'diferencias_divididas': diferencias_divididas
            })
        except ZeroDivisionError as zde:
            return render(request, 'ajuste.html', {
                'fatalerror': "Error: División por cero durante la iteración. Intenta con valores iniciales diferentes."
            })

        except (ValueError, TypeError):
            return render(request, 'ajuste.html', {
                'fatalerror': "No se ha encontrado raíz real en los valores iniciales suministrados."
            })

    return render(request, 'ajuste.html', {
        'fatalerror': "Error Desconocido"
    })

def secanteResuelta(request):
    if request.method == 'POST':
        latex_expr = request.POST.get('expresion-latex')
        print(latex_expr)

        try:
            latex_expr = normalizar_latex(latex_expr)
            print(f"Expresión LaTeX normalizada: {latex_expr}")
            transformations = (standard_transformations + (implicit_multiplication_application,))
            # Parseo de la expresión normalizada
            sympy_expr = parse_expr(latex_expr, transformations=transformations)
            print(f"Expresión SymPy: {sympy_expr}")
            x = symbols('x')

            lista_xi = []
            lista_Fxi = []
            lista_error = []
            error = 0.1

            expresionxi = request.POST.get('x1')
            exprxi = sim.sympify(expresionxi)
            xi = exprxi.evalf()  # Evaluación numérica

            expresionxiAnterior = request.POST.get('x0')
            exprxiAnterior = sim.sympify(expresionxiAnterior)
            xiAnterior = exprxiAnterior.evalf()



            primera = True
            while True:
                if primera:
                    sympy_exprAux = sympy_expr.subs(x, xiAnterior)
                    fxi = N(sympy_exprAux)
                    err = 0
                    lista_error.append(abs(float(err)))
                    lista_xi.append(float(xiAnterior))
                    lista_Fxi.append(float(fxi))
                    err = (xi - xiAnterior) * 100 / xi
                    sympy_exprAux = sympy_expr.subs(x, xi)
                    fxi = N(sympy_exprAux)
                    lista_error.append(abs(float(err)))
                    lista_xi.append(float(xi))
                    lista_Fxi.append(float(fxi))
                    primera = False
                else:
                    sympy_exprAux = sympy_expr.subs(x, xi)
                    fxi = N(sympy_exprAux)
                    sympy_exprAux = sympy_expr.subs(x, xiAnterior)
                    fxiAnterior = N(sympy_exprAux)
                    xinueva = xi - ((fxi) * (xiAnterior - xi) / (fxiAnterior - fxi))
                    xiAnterior = xi
                    xi = xinueva
                    sympy_exprAux = sympy_expr.subs(x, xi)
                    fxi = N(sympy_exprAux)
                    err = (xi - xiAnterior) * 100 / xi
                    lista_error.append(abs(float(err)))
                    lista_xi.append(float(xi))
                    lista_Fxi.append(float(fxi))
                    if abs(err) <= error:
                        break

            valores_c = np.linspace(-100, 400, 20000)
            resultados = [float(N(sympy_expr.subs(x, val))) for val in valores_c]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=valores_c, y=resultados, mode='lines', name='Función'))


            for raices in lista_xi[:-1]:
                sympy_exprAux = sympy_expr.subs(x, raices)
                raiz = float(N(sympy_exprAux))
                fig.add_trace(go.Scatter(x=[raices], y=[raiz], mode='markers', marker=dict(color='green'),
                                         name='Raíz Intermedia'))

                # Agregar la última raíz con color rojo y nombrarla como "Raíz Final"
            ultima_raiz = lista_xi[-1]
            ultima_raiz_valor = float(N(sympy_expr.subs(x, ultima_raiz)))
            fig.add_trace(go.Scatter(x=[ultima_raiz], y=[ultima_raiz_valor], mode='markers', marker=dict(color='red'),
                                     name='Raíz Final'))

            fig.update_layout(
                title='Método de la Secante',
                xaxis_title='Valor de x',
                yaxis_title='Resultado de la ecuación',
                xaxis=dict(
                    range=[-100, 100],
                    zeroline=True,  # Línea del eje en el origen
                    zerolinewidth=2,  # Grosor de la línea del eje x
                    zerolinecolor='red',  # Color de la línea del eje x
                    linecolor='black',  # Color de los bordes del eje x
                    linewidth=2  # Grosor de los bordes del eje x
                ),
                yaxis=dict(
                    range=[-100, 100],
                    zeroline=True,  # Línea del eje en el origen
                    zerolinewidth=2,  # Grosor de la línea del eje y
                    zerolinecolor='blue',  # Color de la línea del eje y
                    linecolor='black',  # Color de los bordes del eje y
                    linewidth=2  # Grosor de los bordes del eje y
                ),
                width=1000,  # Establece el ancho
                height=800
            )

            plot_div = fig.to_html(full_html=False)

            return render(request, 'secanteResuelta.html', {
            'plot_div': plot_div,
            'lista_datos': zip(lista_xi, lista_Fxi, lista_error),'valor_final_x': lista_xi[-1]  # Combina las listas
            })
        except ZeroDivisionError as zde:
            return render(request, 'secante.html', {
                'fatalerror': "Error: División por cero durante la iteración. Intenta con valores iniciales diferentes."
            })

        except (ValueError, TypeError):
            return render(request, 'secante.html', {
                'fatalerror': "No se ha encontrado raíz real en los valores iniciales suministrados."
            })

    return render(request, 'secante.html',{
                'fatalerror': "Error Desconocido"
            })