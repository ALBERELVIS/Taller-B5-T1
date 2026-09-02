# Datos sintéticos para predecir estrés de mercado

**Taller B5-T1** · Contenido de las diapositivas (5 minutos)

> Formato: ocho diapositivas. Cada bloque `---` es una diapositiva. Las rutas de imagen son relativas a la raíz del
> repositorio, así que este fichero se puede maquetar a PDF directamente con Marp, Pandoc o pegando el contenido en
> las plantillas de siempre.

---

## 1 · La pregunta

# ¿Pueden los datos sintéticos mejorar un modelo de ML?

**Problema:** dados 60 días de retornos de 23 activos del S&P 500, predecir si los **próximos 20 días** serán un
**régimen de alta volatilidad** (percentil 95).

**Por qué este problema:** los datos sintéticos solo pueden aportar si los reales son escasos.

- 567 ventanas de estrés en train... pero agrupadas en **14 episodios** en 45 años
- El tamaño muestral **efectivo** de la clase rara es de decenas, no de cientos

**Diseño:** 7 generadores × 5 cantidades de reales × 5 de sintéticos × 3 semillas = **450 experimentos**

---

## 2 · Antes de generar nada: ¿hay señal?

# Nuestro primer target no era predecible

| Target | Predictor | Lift en test | ROC-AUC |
|---|---|---|---|
| Caída ≥ 8% a 20 días | CNN entrenada | **1,09** | **0,494** |
| Régimen de volatilidad | CNN entrenada | **6,61** | **0,827** |

`lift = PR-AUC / tasa base`; el azar da 1.

**No es un bug, es eficiencia débil del mercado:** los retornos futuros no se anticipan desde retornos pasados.
La **volatilidad sí** es persistente.

> Cambiamos el target y conservamos la evidencia. Sin señal en la tarea downstream, todo el estudio mediría ruido.

---

## 3 · El instrumento de medida

# Una arquitectura, congelada

| Modelo | PR-AUC test | Lift |
|---|---|---|
| Azar | 0,078 | 1,00 |
| Regresión logística | 0,282 | 3,62 |
| Baseline: volatilidad realizada | 0,453 | 5,81 |
| **CNN congelada** | **0,515** | **6,61** |

- **CNN > baseline financiero > modelo lineal**
- Métrica: **PR-AUC** (5-13% de positivos: la accuracy es inútil)
- **Sin `class_weight`**: reequilibrar es el trabajo que le pedimos a los sintéticos
- Val y test **100% reales** en las 450 celdas

---

## 4 · Una lección que apareció tres veces

# Una red ReLU no puede calcular una varianza

`Wx + b` + ReLU es una operación de **primer orden**: suma, resta y umbraliza, pero **no multiplica dos entradas**.

Y todo aquí necesita segundos momentos:

| Dónde | Síntoma |
|---|---|
| Clasificador | Perdía contra un baseline de una línea (0,39 vs 0,45) |
| Discriminador de la GAN | No veía que los fakes tenían 5× la volatilidad → GAN divergía |
| Modelo autorregresivo | Tenía que predecir una escala condicional |

**Solución:** concatenar `|x|`. Cero parámetros, no aprende nada, y la entrada sigue siendo una ventana de retornos
crudos → todos los generadores siguen comparables.

---

## 5 · Lo que los generadores no consiguen reproducir

# El factor de mercado

| Generador | std | **Vol. cartera** | **Correlación** | Curtosis |
|---|---|---|---|---|
| **REAL** | 0,0262 | **0,0101** | **0,42** | 47,3 |
| Ruido | 0,0264 | 0,0101 | 0,42 | 46,5 |
| **Factor mercado + idio** | 0,0257 | **0,0102** | **0,38** | 11,2 |
| Gaussiano | 0,0265 | 0,0074 | 0,20 | 3,4 |
| VAE | 0,0228 | 0,0035 | 0,05 | 4,1 |
| GAN | 0,0156 | 0,0018 | 0,03 | 3,8 |

Casi todos aciertan la **escala** y fallan la **volatilidad de cartera**, que es *literalmente* lo que mide el
target.

**Causa:** 23 activos demasiado independientes → al promediarlos, la volatilidad **se diversifica y desaparece**.
El generador de **factor de mercado + idio** ataca ese fallo por construcción: modela `r = β·f + ε` con un AR
para el factor y una gaussiana de Ledoit-Wolf para el residuo.

![](../results/figures/03_diagnostico_distribuciones.png)

---

## 6 · El resultado

# El barrido

![](../results/figures/04_barrido_por_generador.png)

| n_real | Sin sintéticos | Mejor | Ganancia |
|---|---|---|---|
| 500 | 0,480 | 0,523 | **+9,0%** |
| 2 000 | 0,471 | 0,544 | **+15,7%** |
| todos (11 336) | 0,510 | 0,516 | **+1,2%** |

**La ganancia se desvanece con más datos reales.** Ganan los generadores que **reproducen la volatilidad de
cartera** — el trivial de ruido (por copia) y el factor de mercado + idio (por construcción). Los tres generadores
generales entrenables (VAE, GAN, AR) **empeoran** el resultado.

*(Error estándar entre semillas: 0,018)*

---

## 6b · ¿Por qué gana quien reproduce la vol de cartera?

# La fidelidad que importa no es la que parece

Hipótesis: un generador ayuda si reproduce **el estadístico del que depende la etiqueta**, no si parece realista.

| Fidelidad en... | ρ con la ganancia (7 generadores) |
|---|---|
| **Volatilidad de cartera** | **positivo y significativo** |
| Curtosis (colas) | ≈ 0 |

![](../results/figures/05_fidelidad_vs_ganancia.png)

La **t-Student** tiene colas 4× mejores que la gaussiana y **rinde peor**: las colas no definen nuestro target.

Y el **caso extremo** de la hipótesis: el generador de factor de mercado + idio reproduce la volatilidad de cartera
**por construcción** y su ganancia cae exactamente donde la regla la coloca — verificación del mecanismo, no una
coincidencia entre los generadores generales.

> **Lección transferible:** no preguntes si tus sintéticos son realistas. Comprueba si reproducen el estadístico
> del que depende tu etiqueta — o **construye un generador que lo respete por diseño**.

---

## 7 · ¿Y sirve para algo en dinero?

# Backtest de de-risking

**Regla:** si `P(estrés) > umbral` (ajustado en **validación**), mañana en liquidez; si no, cartera.

- Exposición de `t+1` decidida **solo con información hasta `t`**
- Costes de transacción cobrados en cada cambio

| Estrategia | Sharpe | Máx. DD |
|---|---|---|
| Oráculo *(cota superior)* | 1,21 | −15,4% |
| Aleatorio *(control)* | 0,59 | −30,8% |
| 500 reales, sin sintéticos | 0,65 | −21,4% |
| **500 reales + 1000 de ruido** | **0,71** | **−15,4%** |
| Todos + sintéticos | 0,50 | −21,8% |
| Buy & hold | 0,55 | −38,4% |

Con datos escasos: **Sharpe +28% y drawdown de −38% a −15%**, el mismo que el oráculo. Con datos abundantes, nada:
**la misma historia que el PR-AUC**.

![](../results/figures/05_backtest.png)

---

## 8 · Conclusiones

# Qué nos llevamos

1. **Los sintéticos ayudan, condicionalmente.** De **+15,7%** con datos escasos a **+1,2%** con todos, que está
   por debajo del error estándar. Exactamente el patrón anticipado en clase.

2. **Ganan los que reproducen la volatilidad de cartera** — el trivial de ruido por copia y el factor de mercado
   + idio por construcción; los tres generales entrenables empeoran. La razón es medible: con 567 muestras en 14
   episodios, estimar una densidad en 1380 dimensiones es imposible, y lo primero que se pierde es el co-movimiento
   entre activos. El generador factorial lo prueba **construyendo** ese co-movimiento explícitamente.

3. **La regla que se lleva a otros problemas:** no preguntes si tus sintéticos son realistas, comprueba si
   reproducen **el estadístico del que depende tu etiqueta** — o construye un generador que lo respete por diseño.

4. **Limitación honesta:** el test contiene ~5 episodios independientes y el SEM es 0,018. Hablamos de **patrones**,
   no de rankings finos entre generadores.

*Bonus (notebook 06): GAN híbrida cuántica en PennyLane, 8 qubits, comparada contra un gemelo clásico con los
mismos 216 parámetros. El factor limitante no es el generador, es el espacio PCA de 16 dimensiones.*
