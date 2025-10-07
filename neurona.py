import math # Librerbia para el uso de operaciones matematicas simples.

# Clase de iniciacion de la neurona donde se manda la informacion hacia adelante.
class ForwardPass:
    def __init__(self):
        pass

    def sumatoria(self,w,x,b):
        z = (x * w) + b  # w = peso, x = entrada, b = sesgo.
        return z # Retorna el valor de la sumatoria.

    def funcion_sigmoide(self,sumatoria):
        '''
        Funcion de activacion o sigmoide para realizar predicciones sobre el tensor y generar no linedad en el modelo.
        '''
        sigmoide = 1 / (1 + math.e ** -sumatoria) # Exponente del numeto de euler y la sumatoria en negativo, sumanda y dividida al valor 1.
        return sigmoide # Retorno del valor para pasarla a la sigiente clare de perdida.


# Perdida dentro de la entropia cruzada y regularizada.
class PerdidaCalculada:
    def __init__(self,valor_de_la_sigmoide):
        self.sigmoide = valor_de_la_sigmoide # Paso del valor de la funcion sigmoide.
    
    def perdida_entropia_cruzada(self,v_real):
        if v_real == 1: # Si el valor real de la perdida es 1 se hace el logaritmo natural de la sigmoide para calcular la perdida.
            perdida_uno = -math.log(self.sigmoide) 
            return perdida_uno # Retorno del valor de la perdida con valor real de 1.
        elif v_real == 0: # Si el valor real de la perdida es 0 se resta 1 al valor calculado en la sigmoide y se saca su logaritmo natural.
            primer_calculo = (1-self.sigmoide)
            perdida_dos = -math.log(primer_calculo) # Se saca el logaritmo natural del valor negativo.
            return perdida_dos # Retorno del valor de la perdida con valor real de 0.

    def perdida_regularizada(self,peridida_entropia,lamb,peso):
        multiplicacion_potenciada = lamb * (peso ** 2) # Multiplicacion del valor de lambda con el valor del peso potenciado a 2.
        perdida_completa = peridida_entropia + multiplicacion_potenciada # Suma del valor de la perdida con entropia cruzada y la multiplicacion del peso potenciado con el valor de lambda.
        print(f'El valor de la perdida es : {round(perdida_completa,4)}.')
        return perdida_completa

# Retropropagacion para ajustar el error del aprendizaje.
class Backpropagation:
    def __init__(self,valor_de_la_sigmoide,perdida_regularizada): # Paso del la informacion de la sigmoide y la perdida regularizada.
        self.sigmoide = valor_de_la_sigmoide # Valor de la sigmoide calculada en el forward pass.
        self.perdida_completa = perdida_regularizada # Valor de la perdida regularizada calculada en la calse de perdida anterior.

    def derivda_entropica(self,v_real): # Se usa el valor real nuevamente para calcular la derivada de la entropia cruzada.
        if v_real == 1:
            simplificacion_uno = -1/self.sigmoide # derivada calculada con el valor de la sigmoide.
            print(f'La derivada de la entropia cruzada si el valor real es 1 es : {round(simplificacion_uno,4)}')
            return simplificacion_uno
        elif v_real == 2:
            simplificacion_dos = 1 / (1 - self.sigmoide) # derivada calculada si el valor real es 0.
            print(f'La derivada de la entropia cruzada si el valor real es de 0 es : {round(simplificacion_dos,4)}')
            return simplificacion_dos
    
    def derivada_sigmoide(self): # Se calcula la derivada de la sigmoide para usar su calculo junto con la derivada de la entropia cruzada.
        valor_retropropagacion = self.sigmoide * (1  - self.sigmoide) # Formula matematica donde se usa el valor de la sigmoide.
        print(f'El valor de la derivada de la sigmoide es : {round(valor_retropropagacion,4)}')
        return valor_retropropagacion
    
    def gradiente_peso_y_sesgo(self,valor_derivada_entropica,valor_derivada_sigmoide,x,w,lamb):
        '''
        Actualizacion del peso y el sesgo que toma el valor de la derivada de la entropia cruzada y de la sigmoide.
        Junto a esto se usa la entrada , peso y lambda actual.
        '''
        # Calculo del peso
        gradiente_w = (valor_derivada_entropica * valor_derivada_sigmoide * x) + (2 * lamb * w)
        # calculo del sesgo
        gradiente_b = (valor_derivada_entropica * valor_derivada_sigmoide * 1)
        print(f'El gradiente del peso es : {round(gradiente_w,4)}')
        print(f'El gradiente del sesgo es : {round(gradiente_b,4)}')
        return gradiente_b,gradiente_w # Se retornan ambos valores para el calculo del decenso del gradiente.
    
    def descenso_del_gradiente(self,peso_antiguo,sesgo_antiguo,valor_gradiente_w,valor_gradiente_b,alpha):
        '''
        Finalmente se hace el descenso del gradiente con el peso anterior, sesgo anterior y los valores calculados anteriormente para peso y sesgo, alpha es hiperparametro aleatorio.
        '''
        # Actualizacion del peso.
        actualizacion_peso = peso_antiguo - (alpha * valor_gradiente_w)
        # Actualizacion del sesgo.
        actualizacion_sesgo = sesgo_antiguo - (alpha * valor_gradiente_b)
        print(f'El valor de la actualizacion del peso es : {round(actualizacion_peso,4)}')
        print(f'El valor de la actualizacion del sesgo es : {round(actualizacion_sesgo,4)}')
        return actualizacion_peso,actualizacion_sesgo


# Valores de entrada
w = 1.0 # peso
x = 0.8 # entrada
b = -0.5 # sesgo 
a = 0.25 # alpha
v_real = 1  # Valor real
lamb = 0 # Valor de lambda

# Inicio del forwarpass.
calculadora = ForwarPass() # Instancia de la clase fuente.
sum_output = calculadora.sumatoria(w,x,b) # Sumatoria para los siguientes pasos.
sigmoide_output = calculadora.funcion_sigmoide(sum_output) # Instanciar la clase destino, pasando el dato generado como argumento.

#Entrada para calculo de la perdida.
perdida = PerdidaCalculada(sigmoide_output) # Paso de informaciona anterior a la clase hija.
perdida_entropia_cruzada = perdida.perdida_entropia_cruzada(v_real) # Especificar el valor realde la entropia cruzada (No se pasa el valor de la sigmoide ya que este se paso ya a la memoria de la clase).
perdida_regularizada = perdida.perdida_regularizada(perdida_entropia_cruzada,lamb,w) # Especificar el valor de lambda y el peso usado en la sumatoria 

#Paso a la retropropagacion.
retropropagacion = Backpropagation(sigmoide_output,perdida_regularizada) # Intanciar la nueva clase destino con el valor de la sigmoide y perdida regularizada.
derivada_de_entropia = retropropagacion.derivda_entropica(v_real) # Se pasa el valor real la entropia cruzada para usarla en su derivada junto a la sigmoide .
derivada_de_la_sigmoide = retropropagacion.derivada_sigmoide() # Derivada para la sigmoide usando su valor calculado en el forward pass.
# Se instancian ambos resultdos ya que usan los mismos valores como la derivada de la entropia, sigmoide, entrada, peso y lambda.
gradiente_b,gradiente_w = retropropagacion.gradiente_peso_y_sesgo(derivada_de_entropia,derivada_de_la_sigmoide,x,w,lamb) # # se asignan los calculos de ambas derivadas.
# Se realiza el descenso de la gradiente junto a los valores calculados.
x,b = retropropagacion.descenso_del_gradiente(x,b,gradiente_w,gradiente_b,a) # se reasigna el peso nuevo y el sesgo nuevo en el descenso