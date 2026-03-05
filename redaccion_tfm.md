# Redes Neuronales de Grafos para Multigrafos Dirigidos: Aplicación a la Detección de Delitos Financieros

> Redacción elaborada a partir del paper: *"Provably Powerful Graph Neural Networks for Directed Multigraphs"* (Egressy et al., AAAI 2024)

> **Nota**: Este texto ha sido redactado con palabras propias, evitando la traducción literal del paper original. Las ideas y resultados se atribuyen a los autores originales cuando corresponde.

---

## 1. Introducción y Motivación

En los últimos años, las redes neuronales de grafos (GNNs, por sus siglas en inglés) se han consolidado como el paradigma dominante dentro del aprendizaje automático para el tratamiento de datos relacionales. Su capacidad para capturar dependencias estructurales entre entidades las ha convertido en herramientas ampliamente adoptadas en disciplinas tan diversas como la biología molecular, la física de partículas, la predicción meteorológica o el análisis de redes sociales. Sin embargo, cuando estas arquitecturas se aplican al ámbito financiero, y en particular a la detección de actividades ilícitas como el blanqueo de capitales, surgen limitaciones fundamentales que comprometen su eficacia.

Las redes de transacciones financieras presentan dos características intrínsecas que las diferencian de los grafos habitualmente estudiados en la literatura. En primer lugar, las transacciones poseen una dirección inherente: el dinero fluye de una cuenta emisora a una cuenta receptora, lo que convierte la red en un grafo dirigido. En segundo lugar, entre dos mismas cuentas pueden producirse múltiples transacciones a lo largo del tiempo, dando lugar a lo que formalmente se conoce como un multigrafo. La combinación de ambas propiedades define un multigrafo dirigido, una estructura que las GNNs convencionales no están preparadas para manejar de forma adecuada.

El problema se agrava cuando consideramos que los esquemas de blanqueo de capitales se manifiestan típicamente como patrones estructurales específicos dentro de estas redes. Algunos ejemplos bien documentados incluyen los patrones de dispersión-recolección (*scatter-gather*), en los que fondos ilícitos se distribuyen a través de múltiples cuentas intermediarias antes de ser reconsolidados; los ciclos dirigidos, donde el dinero circula en bucles cerrados para dificultar su rastreo; o los bicliques dirigidos, estructuras bipartitas completas que evidencian transferencias sistemáticas entre grupos de cuentas. Las GNNs basadas en paso de mensajes estándar (*Message Passing Neural Networks* o MPNNs) son incapaces de detectar muchos de estos patrones, ya que su capacidad expresiva está acotada por el test de isomorfismo de Weisfeiler-Lehman (WL), que presenta limitaciones conocidas para distinguir determinadas subestructuras.

El trabajo de Egressy et al. (2024) aborda directamente estas carencias proponiendo un conjunto de adaptaciones sencillas pero teóricamente fundamentadas que transforman cualquier MPNN estándar en una arquitectura capaz de operar sobre multigrafos dirigidos con garantías formales de expresividad. Estas adaptaciones son tres: el paso de mensajes inverso (*reverse message passing*), que permite a cada nodo recibir información tanto de sus vecinos entrantes como salientes; la numeración de puertos para multigrafos (*multigraph port numbering*), que asigna identificadores locales a las aristas para distinguir transacciones paralelas entre las mismas cuentas; y los identificadores ego (*ego IDs*), que marcan un nodo central con una característica distintiva para facilitar la detección de ciclos y otras estructuras cerradas.

La principal contribución teórica del paper reside en demostrar que la combinación de estas tres adaptaciones permite detectar cualquier patrón de subgrafo dirigido, algo que ninguna arquitectura previa había logrado. Los autores validan esta capacidad tanto en tareas sintéticas de detección de patrones como en dos problemas reales de análisis de delitos financieros: la detección de transacciones de blanqueo de capitales utilizando datos simulados, y la identificación de cuentas de phishing en la red Ethereum. Los resultados experimentales confirman mejoras sustanciales respecto a las líneas base existentes.

---

## 2. Estado del Arte

### 2.1 Expresividad de las GNNs y el test de Weisfeiler-Lehman

La capacidad expresiva de las redes neuronales de grafos ha sido objeto de un intenso escrutinio teórico durante los últimos años. Un resultado seminal en este campo fue la demostración de Xu et al. (2018) de que las MPNNs estándar tienen un poder discriminativo equivalente, como máximo, al del test de isomorfismo de Weisfeiler-Lehman de primer orden (1-WL). Aunque este test es capaz de distinguir la inmensa mayoría de grafos no isomorfos en la práctica, presenta fallos conocidos ante determinadas configuraciones estructurales. En particular, trabajos como los de Chen et al. (2020) han demostrado que las MPNNs convencionales son incapaces de detectar subestructuras tan fundamentales como los ciclos, una limitación especialmente problemática en el contexto de la detección de fraude financiero.

Esta constatación ha impulsado diversas líneas de investigación orientadas a superar las barreras expresivas del test 1-WL. Una primera aproximación consiste en emular tests de isomorfismo de orden superior, como el k-WL, mediante el paso de mensajes entre k-tuplas de nodos o el uso de modelos tensoriales. Arquitecturas como PPGN (Maron et al., 2019) o los trabajos de Morris et al. (2019) exploran esta dirección, pero su complejidad computacional las hace inviables para aplicaciones a gran escala.

### 2.2 Estrategias para aumentar la expresividad

Una segunda familia de enfoques recurre a la incorporación de características precalculadas que enriquecen la información disponible para la red. Entre las estrategias exploradas se encuentran el conteo de subgrafos (Bouritsas et al., 2022), los embeddings posicionales de nodos (Dwivedi et al., 2021), los identificadores aleatorios (Abboud et al., 2020; Sato et al., 2021) y los identificadores de nodo deterministas (Loukas, 2019). Cada una de estas técnicas aporta información adicional que permite a la red distinguir configuraciones que de otro modo serían indistinguibles.

Una tercera corriente, más reciente, es la de las denominadas *Subgraph GNNs*, que modelan los grafos como colecciones de subgrafos. Dentro de esta categoría se incluyen aproximaciones como DropGNN (Papp et al., 2021), que elimina nodos aleatoriamente y ejecuta la red múltiples veces para recopilar información complementaria, y los grafos anidados de Zhang y Li (2021), que extraen subgrafos locales alrededor de cada nodo. También pertenece a esta familia la arquitectura ID-GNN (You et al., 2021), que introduce los identificadores ego para intentar detectar ciclos. No obstante, Huang et al. (2022) demostraron que la prueba original de detección de ciclos de ID-GNN era incorrecta, y que toda la familia de Subgraph GNNs es incapaz de contar ciclos de longitud superior a cuatro.

### 2.3 GNNs para grafos dirigidos

El estudio de GNNs específicamente diseñadas para grafos dirigidos ha recibido comparativamente menos atención. Los trabajos existentes se centran principalmente en aproximaciones espectrales, como MagNet (Zhang et al., 2021), que propone una red espectral para grafos dirigidos pero cuyo análisis teórico y escalabilidad resultan limitados. Jaume et al. (2019) realizaron una contribución más práctica al proponer la separación de la agregación entre vecinos entrantes y salientes, evitando el tratamiento ingenuo del grafo como no dirigido. Sin embargo, ninguno de estos trabajos ha abordado específicamente el caso de los multigrafos dirigidos, que es precisamente la estructura natural de las redes de transacciones financieras.

### 2.4 GNNs en la detección de fraude financiero

La aplicación de GNNs al análisis de delitos financieros ha experimentado un crecimiento notable. Weber et al. (2019) fueron pioneros en aplicar GNNs estándar a la detección de blanqueo de capitales (*Anti-Money Laundering*, AML), utilizando la red Bitcoin como caso de estudio. Posteriormente, Cardoso et al. (2022) propusieron representar la red de transacciones como un grafo bipartito cuenta-transacción y obtuvieron resultados prometedores en un marco semi-supervisado con su modelo LaundroGraph. En el ámbito de las criptomonedas, se han desarrollado sistemas para la detección de cuentas fraudulentas y de phishing mediante GNNs heterogéneas (Liu et al., 2018; Kanezashi et al., 2022).

No obstante, una limitación común de estos enfoques es que no está claro en qué medida son capaces de detectar los patrones de fraude establecidos en la literatura criminológica. Es precisamente esta brecha la que motiva el trabajo de Egressy et al. (2024), que no solo propone una arquitectura con garantías teóricas de detección de patrones, sino que lo demuestra empíricamente.

---

## 3. Metodología Propuesta

### 3.1 Redes de paso de mensajes (MPNNs)

Las MPNNs constituyen la familia más extendida de redes neuronales de grafos. Su funcionamiento se basa en un mecanismo iterativo de tres pasos que se repite a lo largo de múltiples capas. En cada iteración, cada nodo de la red envía un mensaje con su estado actual a sus vecinos; a continuación, recoge y agrega todos los mensajes recibidos mediante una función invariante a permutaciones (como la suma o el máximo); finalmente, actualiza su propio estado combinando la información recibida con su estado previo. Este proceso permite que la información se propague progresivamente a lo largo de la red, de modo que tras varias capas cada nodo ha acumulado información de vecindarios cada vez más extensos.

En el caso de grafos dirigidos, la agregación convencional solo considera los mensajes procedentes de vecinos entrantes (aquellos que envían transacciones al nodo en cuestión), ignorando por completo la información de los vecinos salientes. Esto supone una pérdida significativa de información estructural.

### 3.2 Paso de mensajes inverso (*Reverse Message Passing*)

La primera adaptación propuesta aborda precisamente esta limitación. En una MPNN estándar dirigida, un nodo no recibe información alguna sobre sus aristas salientes, lo que le impide, por ejemplo, contar cuántas transacciones ha emitido. La solución consiste en implementar dos canales de agregación paralelos: uno para los vecinos entrantes y otro para los vecinos salientes, cada uno con sus propias funciones de agregación y actualización. De este modo, cada nodo recibe una representación completa de su entorno direccional.

Los autores demuestran formalmente que esta adaptación es suficiente para resolver tareas como la determinación del grado de salida de un nodo, algo que las MPNNs estándar no pueden hacer. Además, al mantener los dos canales separados en lugar de simplemente tratar las aristas como no dirigidas, se preserva la información direccional que resultaría crucial para distinguir patrones asimétricos.

### 3.3 Numeración de puertos para multigrafos (*Multigraph Port Numbering*)

La segunda adaptación aborda el problema de las aristas paralelas, es decir, múltiples transacciones entre las mismas dos cuentas. Para detectar patrones como *fan-in* (múltiples vecinos de entrada únicos) o *fan-out* (múltiples vecinos de salida únicos), el modelo necesita distinguir entre aristas que provienen del mismo vecino y aristas de vecinos diferentes.

La técnica de numeración de puertos, originalmente propuesta por Sato et al. (2019) para grafos simples, se adapta aquí a multigrafos dirigidos. Cada arista dirigida recibe dos números de puerto: uno de entrada y otro de salida. Las aristas que comparten origen reciben el mismo número de puerto de salida, y las que comparten destino reciben el mismo número de puerto de entrada. Un aspecto crucial de la propuesta de Egressy et al. es que ambos números de puerto se transmiten como características de la arista en ambas direcciones, de modo que cada nodo conoce tanto el puerto que él asignó a un vecino como el puerto que ese vecino le asignó a él. Esta bidireccionalidad resulta esencial para las garantías teóricas de la arquitectura.

Para romper la simetría en la asignación de puertos, los autores utilizan las marcas temporales de las transacciones como criterio de ordenación, lo cual tiene un significado natural en el contexto financiero: el orden cronológico de las transacciones puede ser relevante para la detección de actividades sospechosas.

### 3.4 Identificadores Ego (*Ego IDs*)

La tercera y última adaptación es la introducción de identificadores ego, una técnica previamente propuesta por You et al. (2021) para la detección de ciclos. La idea consiste en marcar un nodo central con una característica binaria distintiva (el *ego ID*) que lo diferencia del resto de nodos de su vecindario. De este modo, cuando un mensaje recorre un ciclo y regresa al nodo marcado, este puede detectar que la secuencia de mensajes ha completado un circuito cerrado.

Sin embargo, los autores señalan que los ego IDs por sí solos son insuficientes para detectar ciclos de longitud superior a tres. La demostración original de You et al. (2021) contenía un error, como posteriormente confirmaron Huang et al. (2022). La contribución clave de Egressy et al. reside en demostrar que la combinación de las tres adaptaciones --paso de mensajes inverso, numeración de puertos y ego IDs-- sí permite la detección de cualquier patrón de subgrafo en multigrafos dirigidos.

### 3.5 Fundamento teórico: universalidad de la combinación

El resultado teórico principal del paper es el Teorema 4.4, que establece que la combinación de ego IDs, numeración de puertos y paso de mensajes inverso permite asignar identificadores únicos a cada nodo en un multigrafo dirigido conexo. La demostración se basa en mostrar que una GNN puede replicar un algoritmo de etiquetado BFS (búsqueda en anchura) que asigna identificadores únicos capa por capa, partiendo del nodo ego como raíz.

La clave del argumento radica en que nodos activos a la misma distancia del nodo raíz no pueden terminar con el mismo identificador, ya que si dos nodos aceptaran una propuesta del mismo vecino, recibirían diferentes números de puerto (los puertos entrantes son distintos entre sí, los salientes también, y se suma un valor para evitar colisiones entre ambos tipos).

Una vez que cada nodo posee un identificador único, se aplican resultados previos de Loukas (2019) que garantizan que una MPNN suficientemente expresiva con identificadores de nodo únicos es universal, es decir, puede calcular cualquier función sobre el grafo. Como corolario, la arquitectura propuesta puede detectar teóricamente cualquier patrón de subgrafo dirigido.

### 3.6 Complejidad computacional

Un aspecto práctico relevante es que las adaptaciones propuestas no alteran significativamente la complejidad computacional del modelo base. El paso de mensajes inverso multiplica el coste por un factor constante (aproximadamente 2). La numeración de puertos simplemente añade características adicionales a las aristas, sin afectar al tiempo de entrenamiento o inferencia, aunque requiere una precomputación inicial de orden O(m log m), donde m es el número de aristas. Los ego IDs tampoco incrementan la complejidad. En la práctica, la arquitectura Multi-GIN mantiene una velocidad de inferencia superior a 18.000 transacciones por segundo en una sola GPU.

---

## 4. Experimentos y Resultados

### 4.1 Tareas sintéticas de detección de patrones

Para validar las garantías teóricas, los autores diseñaron un banco de pruebas sintético basado en grafos circulantes aleatorios. Este generador permite crear grafos en los que los patrones de interés aparecen de forma natural, evitando el sesgo que introduce la inserción artificial de patrones en grafos preexistentes. Las tareas evaluadas incluyen la detección de grado de entrada y salida, *fan-in* y *fan-out*, ciclos dirigidos de longitud 2 a 6, patrones *scatter-gather* y bicliques dirigidos.

Los resultados confirman con precisión las predicciones teóricas. Las MPNNs estándar obtienen puntuaciones F1 inferiores al 44% en la tarea de grado de salida, mientras que cualquier modelo equipado con paso de mensajes inverso supera el 98%. La numeración de puertos resulta determinante para resolver *fan-in*, y la combinación con paso de mensajes inverso es necesaria para *fan-out*. Los resultados más reveladores se observan en las tareas más complejas: en la detección de *scatter-gather*, la puntuación F1 salta del 67,84% (solo con paso de mensajes inverso y puertos) al 97,42% cuando se añaden los ego IDs. Patrones similares de mejora se observan para ciclos dirigidos y bicliques. La configuración Multi-PNA (con las tres adaptaciones sobre PNA como modelo base) alcanza los mejores resultados globales, con puntuaciones cercanas al 99% en la mayoría de tareas.

### 4.2 Detección de blanqueo de capitales (AML)

Los autores evalúan su propuesta en cuatro conjuntos de datos simulados de transacciones financieras generados con el simulador de IBM (Altman et al., 2023). Estos datasets modelan redes con entre 500.000 y 2,1 millones de cuentas y entre 5 y 32 millones de transacciones, con ratios de transacciones ilícitas que oscilan entre el 0,05% y el 0,11%.

Los resultados demuestran mejoras sustanciales. Sobre el dataset AML Small HI, las adaptaciones elevan la puntuación F1 de la clase minoritaria (transacciones ilícitas) del 28,7% obtenido con GIN estándar al 57,2% con Multi-GIN, una ganancia de casi 30 puntos porcentuales. Las mayores mejoras provienen del paso de mensajes inverso y la numeración de puertos, que conjuntamente elevan la puntuación al 56,9%. Los ego IDs aportan una mejora marginal adicional en estos datasets, probablemente porque los modelos utilizan solo dos capas de GNN, lo que limita la detección de ciclos largos.

Es particularmente significativo que Multi-PNA+EU (PNA con las tres adaptaciones más actualizaciones de arista) supera a todos los baselines en los cuatro datasets AML, incluyendo los métodos basados en árboles de decisión con características de grafo (XGBoost+GFs y LightGBM+GFs), que habían sido el estado del arte previo en aplicaciones financieras. Este resultado es especialmente relevante porque las características manuales de estos baselines están diseñadas específicamente para capturar los patrones de blanqueo utilizados por el simulador, lo que les confiere una ventaja inherente.

El análisis detallado por tipo de patrón revela que la mayoría de transacciones ilícitas pertenecientes a patrones de blanqueo conocidos son correctamente identificadas. Las puntuaciones de recall más bajas se observan en ciclos y bicliques, probablemente debido a la limitación de usar solo dos capas. Sin embargo, las transacciones ilícitas que no pertenecen a ningún patrón específico (*lone transactions*) presentan tasas de recuperación cercanas al 0%, lo que explica las puntuaciones globales más bajas en los datasets con menor proporción de patrones estructurados.

### 4.3 Detección de phishing en Ethereum (ETH)

Para evaluar la propuesta en datos reales, los autores utilizan un dataset de la red Ethereum que contiene 2,9 millones de cuentas y 13 millones de transacciones, con un 0,04% de cuentas etiquetadas como phishing. Este dataset fue enriquecido con características adicionales extraídas directamente de la blockchain, como el nonce, el número de bloque y los costes de gas.

Los resultados siguen la misma tendencia que en AML. La puntuación F1 de Multi-GIN mejora del 26,9% al 42,9%, con el paso de mensajes inverso como la adaptación individual más influyente. Multi-PNA y Multi-PNA+EU superan a todos los baselines por más de 12 puntos porcentuales, alcanzando puntuaciones F1 del 65,3% y 66,6% respectivamente, frente al 53,2% del mejor baseline (LightGBM+GFs).

### 4.4 Resultados en benchmarks adicionales de grafos dirigidos

Para demostrar la aplicabilidad general de las adaptaciones más allá del dominio financiero, los autores también evalúan su propuesta en tres datasets de referencia de grafos dirigidos: Chameleon, Squirrel y Arxiv-Year. Los resultados muestran que las adaptaciones superan al estado del arte (Gradient Gating de Rusch et al., 2022) en dos de los tres benchmarks, con una ganancia de casi 6 puntos porcentuales en el caso de Chameleon.

---

## 5. Conclusiones y Trabajo Futuro

### 5.1 Síntesis de las contribuciones

El trabajo de Egressy et al. (2024) realiza tres contribuciones fundamentales al campo de las redes neuronales de grafos. La primera es de naturaleza teórica: demuestra que la combinación de paso de mensajes inverso, numeración de puertos y ego IDs permite a una MPNN suficientemente expresiva asignar identificadores únicos a todos los nodos de un multigrafo dirigido conexo, lo que implica la capacidad de detectar cualquier patrón de subgrafo. La segunda contribución es empírica: un exhaustivo banco de pruebas sintéticas confirma que estas garantías teóricas se traducen en un rendimiento práctico sobresaliente, con puntuaciones F1 cercanas al 100% en la mayoría de tareas de detección de patrones. La tercera contribución es aplicada: las adaptaciones producen mejoras dramáticas en dos problemas reales de detección de delitos financieros, igualando o superando al estado del arte tanto en datos simulados de blanqueo de capitales como en datos reales de phishing en criptomonedas.

### 5.2 Impacto práctico

Desde una perspectiva práctica, la propuesta tiene implicaciones significativas para el sector financiero. La capacidad de detectar patrones complejos de blanqueo de capitales, como ciclos dirigidos o estructuras de dispersión-recolección, de forma automática y con garantías teóricas, representa un avance considerable respecto a los enfoques basados en reglas o en características manuales que predominan en la industria. Además, el hecho de que las adaptaciones sean modulares y puedan aplicarse sobre cualquier GNN existente facilita enormemente su adopción en sistemas ya desplegados.

La eficiencia computacional de la propuesta también es destacable: la velocidad de inferencia permite procesar decenas de miles de transacciones por segundo, lo que la hace viable para su despliegue en entornos de producción con requisitos de tiempo real.

### 5.3 Limitaciones

No obstante, el trabajo presenta ciertas limitaciones que conviene señalar. En primer lugar, la contribución de los ego IDs en los datasets financieros reales es limitada, lo que sugiere que los patrones más discriminativos en estos contextos son de naturaleza local (grado, fan-in/out) más que global (ciclos largos). Esto podría cambiar con un mayor número de capas, pero a costa de un incremento significativo en el tiempo de ejecución. En segundo lugar, los resultados en AML se obtienen sobre datos simulados, ya que la estricta regulación sobre datos financieros impide el acceso a transacciones reales. Aunque el simulador utiliza patrones de blanqueo establecidos, la brecha entre datos sintéticos y reales sigue siendo una incógnita. En tercer lugar, las transacciones ilícitas que no siguen patrones estructurales reconocibles resultan esencialmente indetectables para el modelo, lo que limita su eficacia ante esquemas de fraude novedosos o atípicos.

### 5.4 Líneas de trabajo futuro

Los autores identifican varias direcciones prometedoras para futuras investigaciones. La exploración de las adaptaciones propuestas en otros dominios que involucren multigrafos dirigidos, como las redes biológicas o los sistemas de transporte, constituye una extensión natural. También resulta de interés el estudio de la relación entre la complejidad computacional de diferentes problemas de detección de subgrafos y el rendimiento de las GNNs, lo que podría guiar el diseño de arquitecturas más eficientes.

### 5.5 Relevancia para este TFM

El trabajo analizado resulta especialmente relevante para esta investigación por varios motivos. En primer lugar, proporciona un marco teórico sólido para entender las limitaciones de las GNNs convencionales en el contexto de la detección de blanqueo de capitales. En segundo lugar, las adaptaciones propuestas ofrecen una solución directamente implementable que no requiere el diseño de arquitecturas completamente nuevas, sino la modificación de modelos existentes. En tercer lugar, los resultados experimentales establecen un punto de referencia actualizado contra el cual evaluar cualquier nueva propuesta en el campo. Finalmente, el análisis detallado por tipos de patrón proporciona una comprensión granular de las fortalezas y debilidades del enfoque, información esencial para guiar el desarrollo de futuros sistemas de detección.

---

## Referencias

- Egressy, B., von Niederhäusern, L., Blanuša, J., Altman, E., Wattenhofer, R., & Atasu, K. (2024). Provably Powerful Graph Neural Networks for Directed Multigraphs. *Proceedings of the AAAI Conference on Artificial Intelligence*.
- Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2018). How powerful are graph neural networks? *arXiv preprint arXiv:1810.00826*.
- Chen, Z., Chen, L., Villar, S., & Bruna, J. (2020). Can graph neural networks count substructures? *Advances in Neural Information Processing Systems*, 33.
- You, J., Gomes-Selman, J. M., Ying, R., & Leskovec, J. (2021). Identity-aware graph neural networks. *Proceedings of the AAAI Conference on Artificial Intelligence*, 35.
- Huang, Y., Peng, X., Ma, J., & Zhang, M. (2022). Boosting the Cycle Counting Power of Graph Neural Networks with I2-GNNs. *arXiv preprint arXiv:2210.13978*.
- Sato, R., Yamada, M., & Kashima, H. (2019). Approximation ratios of graph neural networks for combinatorial problems. *Advances in Neural Information Processing Systems*, 32.
- Jaume, G., et al. (2019). edGNN: a Simple and Powerful GNN for Directed Labeled Graphs. *arXiv preprint arXiv:1904.08745*.
- Weber, M., et al. (2019). Anti-money laundering in bitcoin: Experimenting with graph convolutional networks for financial forensics. *arXiv preprint arXiv:1908.02591*.
- Cardoso, M., Saleiro, P., & Bizarro, P. (2022). LaundroGraph: Self-Supervised Graph Representation Learning for Anti-Money Laundering. *Proceedings of the Third ACM International Conference on AI in Finance*.
- Altman, E., et al. (2023). Realistic Synthetic Financial Transactions for Anti-Money Laundering Models. *NeurIPS Datasets and Benchmarks Track*.
- Loukas, A. (2019). What graph neural networks cannot learn: depth vs width. *arXiv preprint arXiv:1907.03199*.
- Maron, H., et al. (2019). Provably powerful graph networks. *Advances in Neural Information Processing Systems*, 32.
- Morris, C., et al. (2019). Weisfeiler and leman go neural: Higher-order graph neural networks. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33.
