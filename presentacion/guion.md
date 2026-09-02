# Guion de la presentación (5 minutos) y banco de preguntas

## Cómo usar este documento

La primera parte es lo que hay que **decir** en cada diapositiva, con cronometraje. La segunda es un banco de
preguntas probables del profesor con la respuesta preparada, que es donde de verdad se gana o se pierde la nota.

Regla general para los 5 minutos: **no leer las tablas**. Las tablas están en la diapositiva para que él las mire;
tú cuentas la conclusión. Si te quedas sin tiempo, sacrifica la diapositiva 4 (la de ReLU), no la 6 ni la 8.

---

## Cronometraje

| Diapositiva | Tiempo | Acumulado |
|---|---|---|
| 1 · La pregunta | 40 s | 0:40 |
| 2 · ¿Hay señal? | 40 s | 1:20 |
| 3 · El instrumento | 30 s | 1:50 |
| 4 · La lección de ReLU | 25 s | 2:15 |
| 5 · El factor de mercado | 35 s | 2:50 |
| 6 · El barrido | 45 s | 3:35 |
| **6b · Por qué gana quien reproduce la vol de cartera** | **40 s** | **4:15** |
| 7 · Backtest | 25 s | 4:40 |
| 8 · Conclusiones | 20 s | 5:00 |

**Prioridad si vas mal de tiempo.** Las diapositivas 6 y 6b son el resultado y no se tocan. Recorta por este orden:
la 4 (la lección de ReLU) se puede resumir en una frase, la 3 se puede saltar apoyándote en la figura, y la 5 se
puede reducir a la conclusión sin el desarrollo.

---

## 1 · La pregunta · 40 s

> "La pregunta del taller es si los datos sintéticos pueden mejorar un modelo de machine learning. Para poder
> responderla hacía falta un problema donde los datos reales fueran realmente escasos, porque si no, no hay nada que
> ganar y el estudio mide ruido.
>
> Elegí predecir regímenes de alta volatilidad en el S&P 500. Y aquí está el punto importante: en entrenamiento hay
> 567 ventanas de estrés, que parecen muchas. Pero se solapan casi por completo, y agrupadas por episodios de
> mercado son solo **catorce** en cuarenta y cinco años. El tamaño muestral efectivo de la clase rara es de
> decenas, no de cientos. Ese es exactamente el régimen donde los sintéticos deberían aportar."

**No digas** "elegí este problema porque me pareció interesante". Di que lo elegiste porque cumple la condición que
hace medible el efecto.

---

## 2 · ¿Hay señal? · 45 s

> "Antes de generar nada comprobé que la tarea tuviera señal fuera de muestra. Mi primera definición de estrés era
> más intuitiva: una caída de al menos el ocho por ciento en los próximos veinte días. La medí y el lift en test es
> uno coma cero nueve, con un ROC de cero coma cuatro nueve. Es indistinguible del azar.
>
> No es un fallo de implementación: es eficiencia débil del mercado. Los retornos futuros no se anticipan desde
> retornos pasados. La volatilidad, en cambio, es fuertemente persistente, y con el target de régimen de
> volatilidad el mismo modelo pasa a lift seis coma seis.
>
> Cambié el target y me quedé con la evidencia del primero, porque el resultado negativo es informativo por sí
> mismo."

Esta diapositiva es la que demuestra criterio. Cuéntala con calma.

---

## 3 · El instrumento de medida · 35 s

> "El modelo downstream es mi instrumento de medida, así que lo elegí una vez sobre datos cien por cien reales y lo
> congelé. Es la CNN del profesor adaptada a clasificación.
>
> Lo importante es el orden: la CNN supera a los baselines financieros, y estos a la regresión logística. Y los
> baselines no son de paja, la volatilidad realizada es un predictor genuinamente bueno.
>
> Dos decisiones: la métrica es PR-AUC porque con un cinco por ciento de positivos la accuracy no dice nada. Y
> deliberadamente **no** uso pesos de clase, porque reequilibrar la clase minoritaria es justamente el trabajo que
> le estoy pidiendo a los datos sintéticos."

---

## 4 · La lección de ReLU · 35 s

> "Un hallazgo que me apareció tres veces en sitios distintos. Una capa densa o convolucional calcula `Wx + b` y un
> ReLU: es una operación de primer orden, puede sumar y umbralizar, pero no puede multiplicar dos de sus entradas.
> Así que **no puede calcular una varianza**.
>
> Y aquí todo necesita varianzas. Mi primera CNN perdía contra un baseline de una línea. Y el discriminador de la
> GAN no era capaz de ver que las falsificaciones tenían cinco veces la volatilidad correcta, así que el generador
> divergía sin oposición.
>
> La solución es concatenar el valor absoluto de la entrada. Cero parámetros, no aprende nada, y la entrada del
> modelo sigue siendo una ventana de retornos crudos, que es lo que mantiene comparables a todos los generadores."

Si vas justo de tiempo, resume a: *"el discriminador no podía calcular una varianza, así que no veía que los fakes
tenían la volatilidad mal; con `|x|` en la entrada la GAN se estabiliza"*.

---

## 5 · El factor de mercado · 45 s

> "Esta es la tabla que más me enseñó. Casi todos los generadores aciertan la escala de los retornos diarios: es la
> parte fácil. Pero mira la columna de volatilidad de la cartera, que es *literalmente* lo que mide mi target: casi
> todos la fallan por un factor de dos o de cinco.
>
> La causa está en la columna de al lado. Si generas veintitrés activos demasiado independientes, al promediarlos en
> una cartera la volatilidad se diversifica y desaparece. Las ventanas parecen razonables activo por activo y como
> cartera son demasiado tranquilas.
>
> En el gaussiano la causa es concreta: el shrinkage de Ledoit-Wolf que hace posible estimar la covarianza con 567
> muestras en 1380 dimensiones es lo mismo que debilita el factor de mercado. Y se ve también el aviso de clase
> sobre las colas: curtosis tres frente a cuarenta y siete de los datos reales.
>
> La fila del factor de mercado + idio está ahí como respuesta constructiva: modela `r = β·f + ε` con un AR
> pequeño para el factor y una gaussiana de Ledoit-Wolf para el residuo idiosincrático. Es el único generador
> aprendido que recupera la volatilidad de cartera y la correlación media casi exactas, y sirve para verificar
> luego la regla de 6b."

---

## 6 · El barrido · 55 s

Es la diapositiva central. Señala físicamente la gráfica: primero el hueco a la izquierda, luego el cierre a la
derecha.

> "Cada panel es un generador. En negro, la referencia sin sintéticos; en colores, distintas cantidades de datos
> sintéticos.
>
> El patrón es el que anticipaba la clase: **a la izquierda**, con pocos datos reales, hay hueco entre las curvas de
> color y la negra: ahí los sintéticos aportan. **A la derecha**, con todos los datos reales, las curvas se juntan o
> se cruzan: ya no hay nada que aportar y añadirlos puede incluso perjudicar.
>
> Y el resultado que se comenta: ganan los dos generadores que reproducen la volatilidad de cartera, el trivial de
> ruido por copia y el factor de mercado + idio por construcción. Los tres generales entrenables — VAE, GAN, AR —
> quedan por debajo."

**Números para citar de memoria:** con 2000 datos reales, de 0,471 a 0,544, un **+15,7%**. Con todos los datos
reales, de 0,510 a 0,516: un **+1,2%**, por debajo del error estándar de 0,018, es decir, cero. La mayoría de las
mejores configuraciones del barrido son del generador de ruido o del factorial, que son precisamente los dos que
respetan la volatilidad de cartera.

---

## 6b · Por qué gana quien reproduce la vol de cartera · 40 s

Es la diapositiva que convierte un ranking en un hallazgo. Dilo despacio, es el punto más fuerte que tienes.

> "Podría quedarme en 'gana el trivial', pero eso es una anécdota. La pregunta interesante es **por qué**.
>
> Mi hipótesis era que un generador ayuda si reproduce **el estadístico concreto del que depende la etiqueta**, no
> si sus muestras parecen realistas en general. Mi etiqueta se calcula de la volatilidad de la cartera, así que la
> fidelidad *en esa magnitud* debería predecir la ganancia.
>
> Y se puede contrastar. Cruzando los siete generadores, la fidelidad en volatilidad de cartera correlaciona con la
> ganancia con **rho positivo y significativo**. La fidelidad en las colas, en cambio, no predice nada.
>
> El caso que mejor lo ilustra es la t-Student: tiene colas cuatro veces mejores que la gaussiana y **rinde peor**,
> porque las colas no son lo que define mi target.
>
> Por eso gana el generador trivial: al copiar ventanas reales conserva la volatilidad de cartera exactamente. Y
> el caso extremo de la hipótesis es un generador que reproduzca esa magnitud **por construcción**: el de factor de
> mercado más idiosincrasia lo hace, y su ganancia downstream cae exactamente donde la regla la coloca. Es la
> verificación del mecanismo, no una coincidencia entre generadores generales. La lección que me llevo no es 'usa
> ruido', es: **no preguntes si tus sintéticos son realistas, comprueba si reproducen el estadístico del que
> depende tu etiqueta — o construye un generador que lo respete por diseño**."

---

## 7 · Backtest · 25 s

> "Que el PR-AUC mejore no significa que actuar sobre esas predicciones fuera rentable, son preguntas distintas. Así
> que probé la regla más simple: si la probabilidad de estrés supera el umbral ajustado en validación, mañana en
> liquidez.
>
> La exposición de mañana se decide solo con información de hoy, y cobro costes de transacción en cada cambio."

> "Y el resultado cierra el círculo: con datos escasos, añadir sintéticos sube el Sharpe de 0,65 a 0,71 y recorta
> el máximo drawdown del 21 al **15 por ciento**, que es exactamente el drawdown del oráculo. Frente a comprar y
> mantener, 23 puntos menos de caída.
>
> Dos controles para que se lo crean. El aleatorio, con el mismo tiempo invertido, se queda en 0,59 y menos 31: la
> diferencia viene de **cuándo** salgo, no de cuánto. Y con todos los datos reales el beneficio desaparece, igual
> que desaparecía en PR-AUC. **Las dos métricas cuentan la misma historia.**"

**Si preguntan por el retorno:** ninguna estrategia bate el retorno del buy & hold, y es esperable, porque salir
del mercado también te hace perder rebotes. Lo que se compra aquí es riesgo, no rentabilidad, y esa es la métrica
con la que se juzgaría en una mesa real.

---

## 8 · Conclusiones · 20 s

> "Tres cosas. Los sintéticos ayudan de forma **condicional**: un quince por ciento con datos escasos, cero con
> datos abundantes.
>
> Ganan los que reproducen la volatilidad de cartera — el trivial de ruido por copia y el factor de mercado + idio
> por construcción. Los tres generales entrenables empeoran, porque con 567 muestras en 1380 dimensiones lo primero
> que se pierde es el co-movimiento entre activos.
>
> Y la regla que me llevo a otros problemas: **comprueba que tus sintéticos reproducen el estadístico del que
> depende tu etiqueta**, porque el realismo general no basta.
>
> La limitación honesta: mi test tiene unos cinco episodios independientes, así que hablo de patrones y no de
> rankings finos entre generadores."

---
---

# Banco de preguntas del profesor

## Sobre el diseño del problema

**"¿Por qué has cambiado el target a mitad del trabajo? ¿No es hacer trampas elegir el problema que funciona?"**

No, porque el cambio está documentado con la medición que lo motivó y porque el criterio no fue "cuál me da mejor
número" sino "cuál tiene señal fuera de muestra". Sin señal en la tarea downstream, el estudio de datos sintéticos
no puede medir nada: cualquier diferencia sería ruido. Además el resultado negativo se conserva y se reporta, y
tiene una explicación teórica limpia: eficiencia débil del mercado. Lo que sí habría sido cuestionable es probar
varios targets, quedarme con el mejor y no mencionar los otros.

**"El percentil 95 lo has calculado con todos los datos, ¿no?"**

No, solo con el bloque de entrenamiento. Es un estadístico de los datos, así que calcularlo sobre la muestra
completa dejaría que el periodo de test influyera en sus propias etiquetas. Por eso en `data.load_problem` el
etiquetado ocurre **después** de partir, no antes.

**"¿Por qué 60 días de entrada y 20 de horizonte?"**

Los 60 días son los que usa el taller, para que el universo y la ventana sean comparables con sus notebooks. Los 20
días hábiles son un mes de calendario, que es el horizonte natural en gestión de riesgo. No hice análisis de
sensibilidad sobre ninguno de los dos, y lo reconozco como limitación.

## Sobre la partición

**"¿Por qué no has usado `train_test_split` como en mis notebooks?"**

Porque sobre ventanas solapadas mide memoria en vez de predicción. Dos ventanas consecutivas comparten 59 de sus 60
días, así que con una partición aleatoria la del día `t` cae en train y la del día `t+1` en test, y el modelo se
evalúa sobre datos que prácticamente ha visto. Mi partición es cronológica y además descarta 80 ventanas en cada
frontera, porque cada etiqueta mira 20 días al futuro: sin ese embargo, las últimas etiquetas de train estarían
determinadas por días que ya pertenecen a validación.

**"¿Cuánto cambia el resultado con partición aleatoria?"**

No lo cuantifiqué en el barrido completo por tiempo. Lo que sí observé es la señal de alarma típica: con la
partición cronológica, validación y test dan números bastante distintos (val 0,72 frente a test 0,52 en la CNN),
lo que es coherente con que son regímenes de mercado distintos. Una partición aleatoria los acercaría
artificialmente.

## Sobre el PCA y el espacio generativo

**"¿Por qué no has usado PCA, si la covarianza es singular?"**

Empecé con ese plan. Lo medí y el 90% de varianza necesita 986 de 1380 componentes: la curva de varianza explicada
es casi una recta, como la de un ruido blanco. Los retornos diarios no tienen estructura lineal de baja dimensión.
Y truncar el PCA es contraproducente aquí, porque elimina varianza y las ventanas reconstruidas salen más
tranquilas que las reales, que es el sesgo contrario al que necesita un detector de turbulencia. Así que trabajo en
el espacio nativo y resuelvo la singularidad donde le corresponde, dentro del generador gaussiano, con shrinkage de
Ledoit-Wolf, que es el estimador diseñado para `n < p`.

**"¿Y el shrinkage no distorsiona los datos?"**

Sí, y es un compromiso que declaro explícitamente. Ledoit-Wolf empuja la covarianza hacia una identidad escalada, y
eso debilita el factor de mercado: se ve en la tabla, la correlación media baja de 0,42 a 0,20. Pero sin shrinkage
no se puede muestrear en absoluto, porque la matriz no es definida positiva. Es elegir entre un generador
imperfecto y ningún generador.

## Sobre los generadores

**"¿Por qué el modelo de ruido funciona tan bien? ¿No es un poco decepcionante?"**

Es el resultado más interesante del trabajo, precisamente. Y no es casualidad ni algo exclusivo del trivial: al
copiar ventanas reales y perturbarlas, el generador de ruido reproduce **gratis** el factor de mercado, la
volatilidad de cartera y las colas pesadas. Pero **no es el único**: el generador de factor de mercado más
idiosincrasia también gana en varios niveles, y lo hace por el mismo mecanismo — reproducir la volatilidad de
cartera. La diferencia es que él la construye explícitamente en lugar de copiarla. Eso desmonta la lectura "el
trivial gana por casualidad": los dos que ganan son exactamente los dos que respetan la magnitud del target. Los
modelos generales entrenables (VAE, GAN, autorregresivo) no pueden aprender una densidad en 1380 dimensiones a
partir de 567 muestras en 14 episodios independientes, así que lo primero que pierden es el co-movimiento entre
activos, que es lo que aquí importa.

**"¿La GAN ha convergido de verdad?"**

Sí, y la firma es la que usted describió en clase: ninguna de las dos pérdidas se va a cero. Se estabilizan en
`d_loss ≈ 0,90` y `g_loss ≈ 1,29`, así que las dos siguen aprendiendo. Pero llegar ahí me costó dos correcciones:
bajar el learning rate a 1e-4 con beta_1 = 0,5 (las mismas cifras que usted tiene comentadas en la primera línea de
su celda) y, sobre todo, añadir `|x|` a la entrada del discriminador. Sin eso divergía con `d_loss` subiendo a 8.

**"¿Has usado mi truco del ratio de batches?"**

Sí, pero aplicado al número de pasos de gradiente en vez del tamaño del batch, y creo que esa modificación es
necesaria. `train_on_batch` hace **una** actualización sea cual sea el batch, así que darle un batch más grande al
discriminador solo reduce la varianza de su gradiente: no le permite recuperar terreno. Conservo su fórmula, incluido
el `+1` que añadió en directo para que el ratio no divergiera cuando una pérdida llega a cero.

**"El VAE, ¿cómo has evitado el colapso posterior?"**

Con un `beta` muy pequeño, 1e-3, sobre el término KL. Con 1380 dimensiones de salida y 567 muestras, un KL a
intensidad completa colapsa el posterior sobre el prior y el decoder emite la media del dataset para cualquier `z`,
lo cual daría una curva de loss preciosa y un generador inútil. Pero el error más importante que tuve con el VAE fue
otro: al principio devolvía `decoder(z)` como muestra, y eso es la **media** de `p(x|z)`, no una muestra. Generaba
ventanas seis veces más tranquilas que las reales. Hay que volver a sumar la sigma de observación estimada de los
residuos.

**"¿Por qué el autorregresivo no está en el mismo espacio que los demás?"**

Porque las componentes principales de una ventana aplanada no están ordenadas en el tiempo, así que no hay nada
sobre lo que ser autorregresivo. Trabaja en el dominio temporal nativo, que es donde su sesgo inductivo tiene
sentido: modela la escala condicional día a día, es decir aprende explícitamente el agrupamiento de volatilidad, que
es justo la estructura de la que depende mi etiqueta. Sigue produciendo ventanas `(60, 23)`, así que la comparación
downstream no se ve afectada.

## Sobre el experimento

**"¿Cómo sabes que la mejora no es ruido?"**

Tres semillas por celda y reporto error estándar en todas las gráficas. Y digo explícitamente que las diferencias
entre configuraciones cercanas están dentro de ese ruido: mi test tiene unos cinco episodios de estrés
independientes, así que lo que interpreto es la **forma de las curvas** (el hueco a la izquierda que se cierra a la
derecha), que es un patrón consistente entre generadores y semillas, no el ranking entre dos generadores concretos.

**"Añadiendo solo minoritarios rompes la proporción de clases. ¿No invalida eso las métricas?"**

Distorsiona la proporción, sí: con 500 reales y 2000 sintéticos el 82% del entrenamiento es clase positiva. Pero el
PR-AUC es una métrica de **ranking**: no depende de la calibración de la probabilidad, solo del orden en que el
modelo coloca los ejemplos. Es una de las razones por las que la elegí como métrica principal. La precisión y el
recall sí dependen del umbral, y ahí sí hay que leer con cautela; por eso el umbral se ajusta siempre en el conjunto
de validación real y luego se congela.

**"¿Por qué submuestreas al azar en vez de coger los datos más recientes?"**

Porque coger la cola cambiaría el periodo histórico a la vez que el tamaño de muestra, y entonces una caída del
error podría ser escasez de datos o podría ser un régimen de mercado distinto. Submuestrear sobre todo el bloque de
entrenamiento aísla la variable que quiero medir. Mantengo la proporción de clases para que el desbalanceo no sea
otra pieza en movimiento.

## Sobre el bonus cuántico

**"¿Qué hay de cuántico ahí realmente?"**

El núcleo del generador es un circuito variacional: el ruido latente se codifica en ángulos de rotación, se
entrelaza con tres capas de `StronglyEntanglingLayers` y se leen los valores esperados de Pauli-Z de los ocho
qubits. Los parámetros del circuito se entrenan por descenso de gradiente a través del simulador, así que es
entrenable de verdad y no un mapa de características fijo. Todo en `default.qubit` de PennyLane, sin hardware.

**"¿Y gana al clásico?"**

No, y afirmarlo sería sobrevender el resultado. De hecho pierde: 0,300 frente a 0,433 del gemelo clásico, y es 34
veces más lento. Pero los dos empeoran respecto a no usar sintéticos, que es 0,480, así que la lectura correcta no
es "gana el clásico", es "en este espacio ninguno sirve".

Lo que sí hice fue montar la comparación de forma que la pregunta *se pueda* responder: los dos generadores en el
mismo espacio PCA de 16 dimensiones, con el **mismo número exacto de parámetros**, 216, mismo discriminador, mismo
optimizador y mismas semillas. Y el hallazgo útil es que el factor limitante no es el generador: el error de
reconstrucción del propio PCA-16 ya es del orden de un retorno diario, así que hay un techo que ninguno puede
superar. Con ocho qubits en un simulador tampoco hay razón teórica para esperar ventaja.

**"¿Y entonces para qué sirve el bonus?"**

Para una cosa que no esperaba: valida mi resultado principal **fuera de muestra**. La regla de la fidelidad la
estimé con seis generadores que no incluyen a estos dos. Aplicada como predicción, dice que el cuántico debería ir
peor porque reproduce peor la volatilidad de cartera, 0,0070 frente a 0,0114 del clásico. Y acierta el orden. Con
dos puntos no es una demostración, pero es la clase de comprobación que hace pensar que la correlación describe un
mecanismo y no una casualidad entre seis números.

## Preguntas incómodas que conviene tener preparadas

**"¿Cuál es el sesgo de supervivencia de tus 23 activos?"**

Considerable, y lo declaro. Son empresas que existían en 1962 y siguen cotizando, así que están sesgadas hacia
supervivientes. Es el precio de tener 64 años de histórico y varias crisis independientes, que es lo que necesitaba
para el argumento de escasez. Es además el mismo universo que usan sus notebooks.

**"Correlacionar seis puntos y sacar un p-valor es bastante frágil. ¿Te lo crees?"**

Me creo la dirección, no la precisión del p-valor. Ahora hay siete generadores en lugar de seis y la correlación
sigue siendo positiva y significativa, pero tampoco lo presento como prueba definitiva. Lo que le da fuerza no es
el p-valor aislado, sino que va acompañado de cuatro cosas: la hipótesis estaba formulada **antes** de mirar la
correlación y se deriva de cómo construyo la etiqueta, no es pesca de datos; el contraste con la curtosis va en
la dirección predicha, que es el control negativo que esperaba; hay un caso individual que lo ilustra sin
necesidad de estadística, la t-Student con colas cuatro veces mejores que la gaussiana y peor rendimiento; y hay
un caso extremo construido a propósito, el factorial, que reproduce la volatilidad de cartera por diseño y aterriza
exactamente donde la regla predice. Y como salvavidas adicional, el purged K-Fold del notebook 05 muestra que la
forma de las curvas — hueco a la izquierda, cierre a la derecha — sobrevive a cinco cortes cronológicos con
embargo, así que no es un artefacto del único bloque de test del experimento principal.

**"¿Y con el generador factorial no estás haciendo trampa reutilizando el factor de mercado real?"**

No, y es una pregunta legítima. Lo que hago es **modelar** el factor, no copiarlo. La serie del factor de mercado
se estima solo sobre el bloque de entrenamiento y luego se aprende con un autorregresivo pequeño con pérdida NLL
gaussiana; el residuo idiosincrático se ajusta con una gaussiana multivariante con Ledoit-Wolf, también solo con
train. En el muestreo no aparece ninguna ventana real: el factor se genera desde el modelo con burn-in largo y el
residuo se muestrea de la gaussiana ajustada. Además la comprobación TSTR — entrenar la CNN sólo con sintéticos y
evaluar contra reales — mantiene al factorial en el orden que predice la regla de 6b, así que el efecto sobrevive
fuera del régimen exacto donde se estimó, no es un artefacto de reutilización.

**"¿No es circular? El generador de ruido copia datos reales, claro que conserva la volatilidad."**

Es exactamente el mecanismo, sí, pero no es circular: es la explicación. La circularidad existiría si hubiera
definido la fidelidad *después* de ver quién gana, para que encajara. No es el caso: las estadísticas de fidelidad
del notebook 03 se calcularon antes del barrido y son las tres que se usan estándar para validar sintéticos
financieros. Lo que aporta la correlación es que **ordena a los siete**, no solo al de ruido: el gaussiano conserva
un 83% de la volatilidad de cartera y ayuda un poco, la GAN un 24% y perjudica. El factor de mercado + idio la
conserva casi exactamente **por construcción** y también ayuda, en la posición que la regla predice: es la
comprobación de que el mecanismo es real y no un artefacto del ruido copiando ventanas. Y la predicción no trivial
es que la curtosis, que es la estadística que más se cita al validar sintéticos financieros, aquí **no** sirve.

**"Entonces, ¿su conclusión es que no merece la pena entrenar generativos?"**

No, y sería la lectura equivocada. La conclusión es que no merece la pena **en este régimen**: 567 muestras en 1380
dimensiones agrupadas en 14 episodios. Los modelos entrenables necesitan datos para estimar una densidad, y aquí no
los hay; en un problema con datos intradía, donde habría cientos de miles de ventanas, mi propia tabla predice que
el orden se invertiría, porque el cuello de botella dejaría de ser la estimación. Lo que sí generaliza es el
criterio de decisión: mide la fidelidad en el estadístico que define tu etiqueta y usa eso para elegir, en vez de
suponer que el modelo más sofisticado va a ganar.

**"Si tuvieras una semana más, ¿qué harías?"**

Dos cosas, en este orden. Primero, generación condicional completa de pares `X, Y` en vez de solo la clase
minoritaria, que era la cuarta opción de sus diapositivas y que abriría la puerta a corregir sesgos de calibración
además del desbalance. Y segundo, una segunda GAN pensada desde el principio con `|x|` explícitamente en la entrada
del discriminador — la corrección que aquí llegó a mitad del entrenamiento — para ver hasta qué punto el fallo de
la GAN es intrínseco a la familia o simplemente al diseño concreto. El generador factorial que en versiones
anteriores era el primer candidato ya está dentro del trabajo, así que sale de esta lista.

**"¿Qué es lo que más te ha sorprendido?"**

Que el mismo problema conceptual, que una red ReLU no puede calcular un segundo momento, me apareciera tres veces en
tres sitios que parecían no tener nada que ver: el clasificador, el discriminador de la GAN y el modelo
autorregresivo. Y que la solución fuera la misma línea de código en los tres.
