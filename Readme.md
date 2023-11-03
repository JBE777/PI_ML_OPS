<p><img src="formas/henry.png", width="150"></p>
<h5>El primer proyecto individual de la etapa de labs. Este proyecto, PI_ML_OPS, se situá en el rol de un MLOps Engineer y es el:</h5>
<h1 align=center><span style="font-family:Arial Black">PROYECTO INDIVIDUAL Nº1</span></h1>
<h4 align=center><span style="font-family:Arial Black"><i>Cohorte</i></span>: DataPT04</h4>
<h4 align=center><span style="font-family:Arial Black"><i>presentado por</i>:</h4>
<h2 align=center><i>Javier Báez Esqueda</i></h2>

<h2>Objetivo</h2>

>> ***Crear una API con funcionalidades que serán acompañadas por un modelo de Machine Learning, partiendo de tres archivos `gz` que se les extraerá el respectivo archivo `json` para ser convertidos en DataFrames de ser necesario, esto con el fin de generar el dataset de trabajo; este último como elemento para gestar los insumos para las funciones de la API a diseñar.***

<h2>Introducción</h2>

> En este proyecto, trabajamos con tres bases de datos de juegos. Partiendo de los archivos: *`steam_games.json.gz`*,*`user_reviews.json.gz`* y *`users_items.json.gz`* de los cuáles, se extraen los respectivos archivos *`json en folder zip`*: *`steam_games.json`*, *`user_reviews.json`* y *`users_items.json`* que al lograr extraerseles el archivo en cuestión, quedaron así: ***`australian_user_reviews.json`***, ***`australian_users_items.json`*** y ***`output_steam_games.json`***. 
> Estos últimos archivos, una vez abiertos como archivos ***`json`***, se convirtieron a DataFrame. Excepto, este dataset: ***`output_steam_games.json`***, que lo pude abrir directamente con ***`pandas`***. Los DataFrames, se limpiaron y eventualmente se fusionaron para generar los datasets de trabajo.  
> Esto, con la finalidad de ser el insumo para crear las funciones de la ***API*** a diseñar; que a la postre permitiran el diseño de nuestro ***módelo de recomendación***, y cuya representación recaera en la propia función de ***Machine Learning***.

## Desarrollo ETL
### *`Extracción, transformación y carga de datos`*(Pag. 1)
> Para este desarrollo, se crea el Notebook *ETL* y se trabaja con los `archivos json`: **`output_steam_games.json`**, **`australian_user_reviews.json`** y **`australian_users_items.json`**. Para abrir los archivos, importamos las librerias necesarias: *`pandas`*, *`ast`* y *`warnings`*, esta última libreria nos proporciona el método `filterwarnings('ignore')` con el parámetro `ignore` e ignorar los `warnings`, para trabajar con los datasets `json` mencionados. 
> Enseguida, lo que encontramos cuando abrimos cada archivo json. **`output_steam_games.json(1)`**: Abrimos el archivo directamente con `pandas`y eliminamos los valores `nulos`, generando 22530 registros y 13 columnas. Limpiamos la columna `release_date` para extraer la columna `release_year` e incrementar en una columna más. 
> Producto de las limpiezas y transformaciones, se requirio eliminar algunos registros y columnas para reducirse a 4 columnas (`genres`, `title`, `id` y `release_year`) y 22528 registros. 
> La columna `genres` muestra sus registros en forma de `lista`, así que los apilamos en una nueva columna que le llamamos `genero`.Finalmente, eliminamos la columna `genres` que resulta en un dataset de `55607 registros y 4 columnas` que es salvado como: *`df_games.csv`*. **`australian_user_reviews.json(2)`**:
> Apoyado en la librería `ast` y partiendo de una lista vacia, abrimos el archivo mediante `open`, transformamos la lista en un DataFrame que nos arroja 3 columnas:`user_id`,`user_url`,`reviews`. Esta última columna `reviews`, esta anidada; pero, antes de desanidarla crearemos la función  `sentiment_analysis` usando la librería: `TextBlob`, la cuál deriva en 3 métricas: Negative = 0, Neutral = 1 y Positive = 2. Una vez creada la función `sentiment_analysis`, la aplicamos a la columna `reviews` apoyados en el método `apply` es construida la columna:`sentiment_analysis`. 
> Enseguida para desanidar la columna: `reviews`, iteramos las columnas de el archivo(2), usando el método:`iterrows`y
> mediante un diccionario alojamos todas las columnas en el `archivo`(2), incluyendo las columnas desanidadas en una lista que transformamos en DataFrame y eliminamos la columna `reviews`. 
> Ahora, convertimos la columna *`posted`* a formato fecha *`AAAA-MM-DD`*, 
> usando las librerías: `dateutil y parser` y eliminamos la palabra *`Posted`* de cada registro de la columna *`posted`* y aplicamos la función `parse_date` a la columna `posted` y creamos la columna `posted_date` y eliminamos la columna `posted`. Entonces, desde la columna `posted_date` extraemos la columna `posted_year` generando `8 columnas y 59305 registros`. 
> Las columnas generadas son: `user_id`, `user_url`, `sentiment_analysis`, `item_id`, `recommend`, `review`, `posted_date`, `posted_year`. Eliminamos los valores nulos de la columna `posted_year` y las columnas `user_url` y `posted_date` para terminar en `59280 registros y 6 columnas`.
> Resultante de este archivo es salvado como dataset: *`df_reviews.csv`*. **`australian_users_items.json(3)`**:
> Abrimos el archivo mediante `open` y creando un bucle `for` vamos agregando linea por linea en una lista. Transformamos la lista en DataFrame, que nos arroja 5 columnas: `user_id`, `items_count`, `steam_id`, `user_url` e `items`. 
> Esta última columna `items`, esta anidada. La cuál, desanidamos con la iteracion de las columnas de el archivo(3), usando el método:`iterrows`y
> mediante un diccionario alojamos todas las columnas en el `archivo`(3), incluyendo las columnas desanidadas en una lista que transformamos en DataFrame que genera 7 columnas y 5153209 registros. 
> Las columnas generadas son: `user_id`, `item_count`, `steam_id`, `user_url`, `item_id`,`item_name`, `playtime_forever`. 
> Eliminamos las columnas `items_count`, `steam_id`, `user_url` y `item_name` para reducirse a 3 columnas y 5153209 registros.
> Resultante de este archivo es salvado como dataset:*`df_items.csv`*.
### *`Resumen_ETL`*:
> Haciendo las limpiezas y transformaciones correspondientes de los archivos: `df_games.csv`, `df_reviews.csv` y `df_items.csv`, generamos el archivo de trabajo: `df_trabajo.csv` con 11 columnas y 43863 registros. Este archivo es el que utilizamos para trabajar en este proyecto.

## Desarrollo EDA
### *`Exploración y análisis de datos`*(Pag. 2)
> Este desarrollo, se crea el Notebook *EDA* y se trabaja con el `archivo.csv`: **`df_trabajo.csv`**. Para abrir los archivos, 
importamos la libreria necesarias: *`pandas, seaborn, matplotlib.pyplot, wordcloud, STOPWORDS y warnings`*, esta última libreria nos proporciona el método `filterwarnings('ignore')` con el parámetro `ignore` e ignorar los `warnings`, que nos sirve para construir los insumos de cada función y salvando, el respectivo archivo como `csv`. **`Creando el insumo de mis endpoints`**:
> Segun el método `info()`, tenemos un total de 43863 registros y 11 columnas en nuestro dataset de trabajo, 6 variables son numéricas (`item_id`, `playtime_forever`, `sentiment_analysis`, `posted_year`, `id` y `release_year`), 
1 variable es booleana (`recommend`) y 4 variables son objetos(`user_id`,`review`,`title` y `genero`). Sin embargo, necesitamos generar la columna `playtime_hours` a partir de la columna `playtime_forever`, y con ésto poder completar la totalidad de columnas necesitadas para crear los endpointsa continuación .
`1)`**`df[['genero','release_year','playtime_hours']]`**, genera el archivo `genero.csv`. `2)`**`df[['genero','posted_year','user_id','playtime_hours']]`**, genera el archivo `userforgenre.csv`. 
`3)`**`df[['recommend','posted_year','sentiment_analysis','title']]`**, genera el archivo `UsersRecommend.csv`. 
`4)`**`df[['recommend','posted_year','sentiment_analysis','title']]`**, genera el archivo `UsersNotRecommend.csv`. 
`5)`**`df[['release_year','review','sentiment_analysis']]`**, genera el archivo `sentimientos.csv`.`6)`**`df[['title','review']]`**, genera el archivo `recomendacion_juego.csv`. 
`7)`**`df[['user_id','title']]`**, genera el archivo `recomendacion_usuario.csv`. **`En el análisis de los datos:`** Empezamos a trabajar con las variables numéricas (`item_id, id, sentiment_analysis, posted_year, release_year, playtime_hours`) aplicando los metodos necesarios para obtener los
estadísticos `describe()` y las correlaciones `corr()` de nuestras variables sobresaliendo de los `3` parámetros de la columna de *`sentiment_analysis`*, el parámetro *`Neutral`* con *`27913`* registros
de los `43863` registros totales. El año de los `reviews`, la columna *`posted_year`* sobresale el año *`2014`* con *`15680`* registros, y seguido muy cerca por el año *`2015`* con `14311` registros.
La columna *`release_year`* muestra *`15684`* registros para el año *`2017`*, y en segundo lugar el año `2016` con `10996` registros. La correlación de Pearson más alta es `0.580668` y esta entre las columnas **`release_year`** y **`id`**. Enseguida, usamos la variable `booleana` representada por la columna *`recommend`*
para matizar las variables más correlacionadas mediante *`True`* y *`False`* sobresaliendo *`True`* con *`38926`* registros y `False` con `4937` registros siendo considerados
respectivamente los juegos *`Recomendados`* y *`No Recomendados`*. En cantidad de títulos más recomendados de videojuegos recae en **`Lost Summoner Kitty, Real Pool 3D - Poolians y Caviar - Endless Stress Reliever`** con *`5`* registros en cantidad.
El segundo lugar con *`4`* registros esta representado por **`Ironbound`** y finalmente el tercer lugar con *`3`* registros lo representa **`Battle Royale Trainer`**.
Para continuar con los titulos recomendados, graficamos una nube de palabras y encontramos que las palabras que más se repiten son: **`Collector, Ultimate, Puzzles, Puzzle, Pack, War, Original`**. 
Analizando los géneros de los videojuegos, vemos la primera posición la ocupa `Indie` con `10109` registros y `Action` le sigue con `7378` registros.
Las palabras o géneros que mas se repiten en su nube de palabras son: `Indie`, `Action`, `Strategy` y `RPG`. Y en la nube de palabras de la columna `review`, podemos ver las palabras `game, play, fun, one, will, good, time, make.`
### *`Resumen_EDA`*:
>En esencia los `títulos mas recomendados` con `género` y `tipo de comentario son:`1) **`Lost Summoner Kitty`**: `Casual[0], Action[1], Indie[2], Strategy[3] y Simulation[5]` con comentario `Neutral`. 
2)**`Real Pool 3D - Poolians`**: `Free to Play[6], Indie[7], Casual[9], Sports[12] y Simulation[16]` con comentario `Positive`. 
3)**`Caviar - Endless Stress Reliever`**: `Adventure[24], Adventure[26], Action[18791]` con comentario `Positive`. 4)**`Ironbound`**: `Free to Play[4], Indie[8], Strategy[10] y RPG[11]` con comentarios `Neutral` para el `género:Free to Play` y `Positive` para el resto de `géneros `.  
5)**`Battle Royale Trainer`**: `Action[13], Adventure[17] y Simulation[20]` con comentarios `Positive` para el `género:Action` y `Neutral` para el resto de `géneros `. 
  
## Desarrollo API
### *`Funciones API`*(Pag. 3)
> Para este desarrollo, se crea el Notebook *`Funciones API`* y se trabaja con los `archivos csv`como insumo para crear las respectivas `funciones API`. Para eso, se importan las librerias necesarias: *`pandas`*, *`from sklearn.feature_extraction.text import TfidfVectorizer`* y *`from sklearn.metrics.pairwise import linear_kernel`*. 
>  - Con el archivo:`genero.csv`, creamos la función:**`def PlayTimeGenre`**(`genero`:str):      
>  - Con el archivo:`userforgenre.csv`, creamos la función:**`def UserForGenre`**(`genero`:str):
>  - Con el archivo:`UsersRecommend.csv`, creamos la función:**`def UsersRecommend`**(`anio`:int): 
>  - Con el archivo:`UsersNotRecommend.csv`, creamos la función:**`def UsersNotRecommend`**(`anio`:int): 
>  - Con el archivo:`sentimientos.csv`, creamos la función:**`def sentiment_analysis`**(`anio`:int): 
>  - Con el archivo:`recomendacion_juego.csv`, creamos la función:**`def recomendacion_juego`**(`titulo`): 
>  - Con el archivo:`recomendacion_usuario.csv`, creamos la función:**`def recomendacion_usuario`**(`user_id`): 

### *Integración Funciones API*: *`main.py`*
> Para integrar las funciones API, es necesario acceder a `VSCode`. En mi caso, yo usé `Anaconda-Environments-myenv-Open Terminal`.
Una vez en `terminal`, teclea: **cd Desktop** `enter`, **cd PI_ML_OPS** `enter`, **pip install fastapi** `enter`, **pip install uvicorn** `enter`. 
En la misma ruta de `terminal de anaconda`, teclea separado: **code .** `enter` para entrar a VSCode. Una vez en VSCode, creamos el archivo:**main.py** para incluir primeramente, las librerias necesarias: 
**`importar pandas as pd`**, **`from sklearn.feature_extraction.text import TfidfVectorizer`**,
**`from sklearn.metrics.pairwise import linear_kernel`** y **`from fastapi import FastAPI`** 
>> Escribimos tambien: `app = FastAPI`()
    
### *Cargamos las funciones con sus respectivos decoradores:* 
    
1. La primera función debe devolver: Año con más horas jugadas para dicho género. Ejemplo de retorno: 
    {"Año de lanzamiento con más horas jugadas para el Género X" : 2013} con Input:genero.
    - `@app.get`('/**`PlayTimeGenre`**/{`genero`}')
    - **`def PlayTimeGenre`**(`genero`):
2. La segunda función debe devolver: El usuario que acumula más horas jugadas para el género dado y una 
    lista de la acumulación de las horas jugadas por año. Ejemplo de retorno: 
    {"Usuario con más horas jugadas para el Género X" : us213ndjss09sdf, "Horas jugadas":
    [{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]} con Input:genero
    - `@app.get`('/**`UserForGenre`**/{`genero`}')
    - **`def UserForGenre`**(`genero`): 
3. La tercera función debe devolver: El top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = 
    True y comentarios positivos/neutrales). Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
    con Input:anio=Año
    - `@app.get`('/**`UsersRecommend`**/{`anio`}')
    - **`def UsersRecommend`**(`anio`): 
4. La cuarta función debe devolver: El top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = 
    False y comentarios negativos). Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
    con Input:anio=Año 
    - `@app.get`('/**`UsersNotRecommend`**/{`anio`}')
    - **`def UsersNotRecommend`**(`anio`): 
5. La quinta función debe devolver: Según el año de lanzamiento, una lista con la cantidad de registros de reseñas
       de usuarios que se encuentren categorizados con un análisis de sentimiento. Con Input:anio=Año 
    - `@app.get`('/**`sentiment_analysis`**/{`anio`}')
    - **`def sentiment_analysis`**(`anio`):  
6. La sexta función debe devolver: Ingresando el titulo de un producto, deberíamos recibir una lista con 5 juegos 
        recomendados similares al ingresado.
    - `@app.get`('/**`recomendacion_juego`**/{`titulo`}')
    - **`def recomendacion_juego`**(`titulo`):   
7. La septima función debe devolver: Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.
    - `@app.get`('/**`recomendacion_usuario`**/{`user_id`}')
    - **`def recomendacion_usuario`**(`user_id`):
    
### **`Deployment`**: 
 - Para que la API pueda ser consumida localmente, y una vez hecho lo anterior, es nevesario teclear desde terminal de `VSCode` o
desde la terminal donde accediste a tu proyecto. El siguiente comando: **uvicorn main:app --reload** `enter` para obtener  en tu localhost:http://127.0.0.1:8000/docs.
 - Una vez, probada la API localmente puedes continuar en `VSCode` para agregar el archivo txt: `requirements.txt` creado vacío y agregando solamente las librerias que se crean convenientes. 
 - Enseguida subí mi proyecto a GitHub, en mi caso y desde terminal de VSCode, escribí los comandos: **git add .**`enter`, **git commit -m 'PI_ML_OPS'** `enter` y **git push** `enter`.
 - Después accedí a mi cuenta de GitHub, abriendo mis repositorios para ubicar mi proyecto: **PI_ML_OPS** para abrirlo e inspeccionar que todo estaba bien.
 - Finalmente entramos a: https://render.com, y conectamos con la cuenta GitHub para acceder a un `WebService` para cargar el proyecto, para que la `API` este disponible y consumible en la `Nube`, que si todo esta bien: 
 el icono de color verde `live` aparece y en todos los casos aparentemente se puede ver el link de tu API. En mi caso: https://fastapi-cb8b.onrender.com
                                         
## **Sistema de recomendación**
Una vez creados los insumos de las funciones de recomendación, las cuáles son representadas por: def <b>recomendacion_juego(<em>`id_producto`</em>)</b> y def <b>recomendacion_usuario(<em>`user_id`</em>)</b>. Creamos las respectivas funciones de recomendación basadas en los siguientes datasets: <i>recomendacion_juego.csv</i> y <i>recomendacion_usuario.csv</i> para integrarlas en la lista de funciones creadas en el
notebook: <em>Funciones_API</em>

<h3>Acceso Repositorio GitHub</h3>

https://github.com/JBE777/PI_ML_OPS

<h3>Acceso Deploy Render</h3>

https://fastapi-cb8b.onrender.com/docs#/
                                                                                                                     
<h3>Acceso Video</h3>
<p><img src="formas/Git.png", width="150"></p>
<h6 align=right>México - 2023</h6>