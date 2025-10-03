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
        return perdida_completa