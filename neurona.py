import math


class ForwarPass:
    def __init__(self):
        pass

    def sumatoria(self,w,x,b):
        z = (x * w) + b
        return z

    def funcion_sigmoide(self,sumatoria):
        sigmoide = 1 / (1 + math.e ** -sumatoria)
        return sigmoide


class PerdidaCalculada:
    def __init__(self,valor_de_la_sigmoide):
        self.sigmoide = valor_de_la_sigmoide
    
    def perdida_entropia_cruzada(self,v_real):
        if v_real == 1:
            perdida_uno = -math.log(self.sigmoide)
            return perdida_uno
        elif v_real == 0:
            primer_calculo = (1-self.sigmoide)
            perdida_dos = -math.log(primer_calculo)
            return perdida_dos

    def perdida_regularizada(self,peridida_entropia,lamb,peso):
        multiplicacion_potenciada = lamb * (peso ** 2)
        perdida_completa = peridida_entropia + multiplicacion_potenciada
        print(f'El valor de la perdida es : {round(perdida_completa,4)}.')
        return perdida_completa

class Backpropagation:
    def __init__(self,valor_de_la_sigmoide,perdida_regularizada):
        self.sigmoide = valor_de_la_sigmoide
        self.perdida_completa = perdida_regularizada

    def derivda_entropica(self,v_real):
        if v_real == 1:
            simplificacion_uno = -1/self.sigmoide
            print(f'La derivada de la entropia cruzada si el valor real es 1 es : {round(simplificacion_uno,4)}')
            return simplificacion_uno
        elif v_real == 2:
            simplificacion_dos = 1 / (1 - self.sigmoide)
            print(f'La derivada de la entropia cruzada si el valor real es de 0 es : {round(simplificacion_dos,4)}')
            return simplificacion_dos
    
    def derivada_sigmoide(self):
        valor_retropropagacion = self.sigmoide * (1  - self.sigmoide)
        print(f'El valor de la derivada de la sigmoide es : {round(valor_retropropagacion,4)}')
        return valor_retropropagacion
    
    def gradiente_peso_y_sesgo(self,valor_derivada_entropica,valor_derivada_sigmoide,x,w,lamb):
        gradiente_w = (valor_derivada_entropica * valor_derivada_sigmoide * x) + (2 * lamb * w)
        gradiente_b = (valor_derivada_entropica * valor_derivada_sigmoide * 1)
        print(f'El gradiente del peso es : {round(gradiente_w,4)}')
        print(f'El gradiente del sesgo es : {round(gradiente_b,4)}')
        return gradiente_b,gradiente_w
    
    def descenso_del_gradiente(self,peso_antiguo,sesgo_antiguo,valor_gradiente_w,valor_gradiente_b,alpha):
        actualizacion_peso = peso_antiguo - (alpha * valor_gradiente_w)
        actualizacion_sesgo = sesgo_antiguo - (alpha * valor_gradiente_b)
        print(f'El valor de la actualizacion del peso es : {round(actualizacion_peso,4)}')
        print(f'El valor de la actualizacion del sesgo es : {round(actualizacion_sesgo,4)}')
        return actualizacion_peso,actualizacion_sesgo


