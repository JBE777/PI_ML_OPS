<p><img src="formas/henry.png", width="150"></p>
<h5>El primer proyecto individual de la etapa de labs. Este proyecto, PI_ML_OPS, se situá en el rol de un MLOps Engineer y es el:</h5>
<h1 align=center><span style="font-family:Arial Black">PROYECTO INDIVIDUAL Nº1</span></h1>
<h6 align=center><i>de</i></h6>
<h4 align=center><i>Cohorte</i>: DataPT04</h4>
<h4 align=center><i>es presentado por</i>:</h4>
<h2 align=center><i>Javier Báez Esqueda</i></h2>

<h2>Objetivo</h2>

> ***Crear una API con funcionalidades que serán acompañadas por un modelo de Machine Learning, partiendo de tres archivos `gz` que se les extraerá el respectivo archivo `json` para ser convertidos en DataFrames de ser necesario, esto con el fin de generar el dataset de trabajo; este último como elemento para gestar los insumos para las funciones de la API a diseñar.***

<h2>Introducción</h2>

> En este proyecto, trabajamos con tres bases de datos de juegos. Partiendo de los archivos: *`steam_games.json.gz`*,*`user_reviews.json.gz`* y *`users_items.json.gz`* de los cuáles, se extraen los respectivos archivos *`json en folder zip`*: *`steam_games.json`*, *`user_reviews.json`* y *`users_items.json`* que al lograr extraerseles el archivo en cuestión, quedaron así: ***`australian_user_reviews.json`***, ***`australian_users_items.json`*** y ***`output_steam_games.json`***. 
> Estos últimos archivos, una vez abiertos como archivos ***`json`***, se convirtieron a DataFrame. Excepto, este dataset: ***`output_steam_games.json`***, que lo pude abrir directamente con ***`pandas`***. Los DataFrames, se limpiaron y eventualmente se fusionaron para generar los datasets de trabajo.  
> Esto, con la finalidad de ser el insumo para crear las funciones de la ***API*** a diseñar; que a la postre permitiran el diseño de nuestro ***módelo de recomendación***, y cuya representación recaera en la propia función de ***Machine Learning***.

## Desarrollo ETL
### *`Extracción, transformación y carga de datos`*(Pag. 1)
> Para este desarrollo, se crea el Notebook *ETL* y se trabaja con los `archivos json`: ***output_steam_games.json***, ***australian_user_reviews.json*** y ***australian_users_items.json***. Para abrir los archivos, importamos las librerias necesarias: ***pandas*** y ***ast*** para trabajar con los datasets `json` mencionados. 
> Enseguida, lo que encontramos cuando abrimos cada archivo json.

>  *`output_steam_games.json(1)`*:
> Abrimos el archivo directamente con `pandas`y eliminamos los valores `nulos`, generando 22530 registros y 13 columnas. 
> Extraemos la columna `year` de la columna `release_date`, para incrementar en una columna más. 
> Eliminamos las columnas para reducir a 6 columnas: `genres`, `app_name`, `title`, `release_date`, `id` y `year`. 
> La columna `genres` muestra sus registros en forma de `lista`, así que los apilamos en una nueva columna que le llamamos `genero`.
> Finalmente, eliminamos la columna `genres` que resulta en un dataset de `55607 registros y 6 columnas` que es salvado como: ***`df_games.csv`***

>  *`australian_user_reviews.json(2)`*:
> Apoyado en la librería `ast` y partiendo de una lista vacia, abrimos el archivo mediante `open`, transformamos la lista en un DataFrame que nos arroja 3 columnas:`user_id`,`user_url`,`reviews`. Esta última columna `reviews`, esta anidada; pero, antes de desanidarla crearemos la función  `sentiment_analysis` usando la librería: `TextBlob`, la cuál deriva en 3 métricas: Negative = 0, Neutral = 1 y Positive = 2. Una vez creada la función `sentiment_analysis`, la aplicamos a la columna `reviews` apoyados en el método `apply` es construida la columna:`sentiment_analysis`. 
> Enseguida para desanidar la columna: `reviews`, iteramos las columnas de el archivo(2), usando el método:`iterrows`y
> mediante un diccionario alojamos todas las columnas en el `archivo`(2), incluyendo las columnas desanidadas en una lista que transformamos en DataFrame y eliminamos la columna `reviews`. 
> Ahora, convertimos la columna *`posted`* a formato fecha *`AAAA-MM-DD`*, 
> usando las librerías: `dateutil y parser` y eliminamos la palabra *`Posted`* de cada registro de la columna *`posted`* y aplicamos la función `parse_date`a la columna `posted` y creamos la columna `posted_date` y eliminamos la columna `posted`. Entonces, desde la columna `posted_date` extraemos la columna `year` generando `8 columnas y 59305 registros`. 
> Las columnas generadas son: `user_id`, `user_url`, `sentiment_analysis`, `item_id`, `recommend`, `review`, `posted_date`, `year`. Eliminamos los valores nulos de la columna `year` para terminar con `59280 registros y 8 columnas`.
> Resultante de este archivo es salvado como dataset: ***`df_reviews.csv`***.

>  *`australian_users_items.json(3)`*:
> Abrimos el archivo mediante `open` y creando un bucle `for` vamos agregando linea por linea en una lista. Transformamos la lista en DataFrame, que nos arroja 5 columnas: `user_id`, `items_count`, `steam_id`, `user_url` e `items`. 
> Esta última columna `items`, esta anidada. La cuál, desanidamos con la iteracion de las columnas de el archivo(3), usando el método:`iterrows`y
> mediante un diccionario alojamos todas las columnas en el `archivo`(3), incluyendo las columnas desanidadas en una lista que transformamos en DataFrame que genera 7 columnas y 5153209 registros. 
> Las columnas generadas son: `user_id`, `item_count`, `steam_id`, `user_url`, `item_id`,`item_name`, `playtime_forever`. 
> Eliminamos la columna `steam_id` y los registros duplicados para tomar para el proyecto solamente los últimos `80000 registros y 6 columnas`.
> Resultante de este archivo es salvado como dataset: ***`df_items.csv`***. Sí tomamos todo el dataset con `5094092 registros y 6 columnas`, podemos salvarlo como:  ***`df_items_plus.csv`***

### *`Resumen_ETL`*:
De la suma de los archivos: `df_games.csv`, `df_reviews.csv` y `df_items_plus.csv`, y antes de renombrar la columna `year` de los 2 primeros archivos
como `release_year` y `posted_year` respectivamente, generamos el archivo de trabajo: `df_trabajo.csv` con 11 columnas y 42454 registros. Este archivo es que utilizamos para trabajar
porque la fusión de los archivos `df_games.csv`, `df_reviews.csv` y `df_itema.csv` resultaron únicamente con las mismas 11 columnas, pero con sólo 890 registros.

## Desarrollo EDA
### *`Exploración y análisis de datos`*(Pag. 2)
> Para este desarrollo, se crea el Notebook *EDA* y se trabaja con el `archivo.csv`: **`df_trabajo.csv`**. Para abrir los archivos, importamos la libreria necesaria: ***pandas*** para trabajar con los datasets mencionados. Construyendo los insumos de cada función y salvando, el respectivo archivo como `csv`. 
### *`Analizando los datos para crear el insumo de mis endpoints`*.
Para esto, necesitamos revisar que columnas de nuestro archivo de trabajo nos genera que archivo.csv o insumo para nuestro endpoint.
* Las columnas: `df[['genero','release_year','playtime_forever']]`, genera el archivo `genero.csv`.
* Las columnas: `df[['genero','posted_year','user_id','playtime_forever']]`, genera el archivo `userforgenre.csv`.
* Las columnas: `df[['recommend','posted_year','sentiment_analysis','title']]`, genera el archivo `UsersRecommend.csv`.
* Las columnas: `df[['recommend','posted_year','sentiment_analysis','title']]`, genera el archivo `UsersNotRecommend.csv`.
* Las columnas: `df[['release_year','review','sentiment_analysis']]`, genera el archivo `sentimientos.csv`.
* Las columnas: `df[['title','review']]`, genera el archivo `recomendacion_juego.csv`.
* Las columnas: `df[['user_id','title']]`, genera el archivo `recomendacion_usuario.csv`.
### *`Analizando los datos de mis variables`*
Segun el método `info()`, tenemos un total de  42454 registros y 11 columnas en nuestro dataset de trabajo, 6 variables son numéricas (`item_id`, `playtime_forever`, `sentiment_analysis`, `posted_year`, `id` y `release_year`), 
1 variable es booleana (`recommend`) y 4 variables son objetos(`user_id`,`review`,`title` y `genero`).        
 
## Desarrollo API
### *Funciones API*: *`(Pag. 3)`*
> Para este desarrollo, se crea el Notebook *Funciones API* y se trabaja con los `archivos csv`como insumo para crear las respectivas `funciones API`. Para eso, se importan las librerias necesarias: *`pandas`*, *`from sklearn.feature_extraction.text import TfidfVectorizer`* y *`from sklearn.metrics.pairwise import linear_kernel`*. 

>* Con los archivos:`user.csv` y `price.csv`, creamos la función:**def userdata**(`User_id`:str):      
>* Con los archivos:`countreviews.csv`, creamos la función:**def countreviews**(`YYYY-MM-DD` y `YYYY-MM-DD`:str):
>* Con los archivos:`genero.csv`, creamos la función:**def genre**(`genero`:str): 
>* Con los archivos:`userforgenre.csv`, creamos la función:**def userforgenre**(`genero`:str): 
>* Con los archivos:`developer.csv`, creamos la función:**def developer**(`desarrollador`:str): 
>* Con los archivos:`sentiments.csv`, creamos la función:**def sentiment_analysis**(`anio`:int): 
>* Con los archivos:`recomendacion_producto.csv`, creamos la función:**def recomendacion_producto**(`titulo`:str): 

### *Integración Funciones API*: *`main.py`*
> Para integrar las funciones API, es necesario acceder a `VSCode`. En mi caso, yo usé `Anaconda-Environments-myenv-Open Terminal`.
Una vez en `terminal`, teclea: **cd Desktop** `enter`, **cd PI_ML_OPS** `enter`, **pip install fastapi** `enter`, **pip install uvicorn** `enter`. 
En la misma ruta de `terminal de anaconda`, teclea separado: **code .** `enter` para entrar a VSCode. Una vez en VSCode, creamos el archivo:**main.py** para incluir primeramente, las librerias necesarias: 
**`importar pandas as pd`**, **`from sklearn.feature_extraction.text import TfidfVectorizer`**,
**`from sklearn.metrics.pairwise import linear_kernel`** y **`from fastapi import FastAPI`** 
>> Escribimos tambien: `app = FastAPI`()
    
### *Cargamos las funciones con sus respectivos decoradores:* 
    
1. La primera función debe devolver: Cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.
    - `@app.get`('/*userdata*/{`User_id`}')
    - **def userdata**(`User_id`):
2. La segunda función debe devolver: Cantidad de usuarios que realizaron reviews entre las fechas dadas y, el porcentaje de recomendación de los mismos en base a reviews.recommend.
    - `@app.get`('/*countreviews*/{`date1, date2`}')
    - **def countreviews**(`date1, date2`): 
3. La tercera función debe devolver: Puesto en el que se encuentra un género sobre el ranking de los mismos y analizados bajo la columna PlayTimeForever.
    - `@app.get`('/*genre*/{`genero`}')
    - **def genre**(`genero`): 
4. La cuarta función debe devolver: Top 5 de usuarios con más horas de juego en el género dado, con su URL (del user) y user_id.
    - `@app.get`('/*userforgenre*/{`genero`}')
    - **def userforgenre**(`genero`): 
5. La quinta función debe devolver: Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.
    - `@app.get`('/*developer*/{`desarrollador`}')
    - **def developer**(`desarrollador`):  
6. La sexta función debe devolver: Según el año de lanzamiento, una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
    - `@app.get`('/*sentiment_analysis*/{`anio`}')
    - **def sentiment_analysis**(`anio`):   
7. La septima función debe devolver: Ingresando el titulo de producto, una lista con 5 juegos recomendados para dicho producto.
    - `@app.get`('/*recomendacion_producto*/{`titulo`}')
    - **def recomendacion_producto**(`titulo`):
    
### **`Deployment`**: 
* Para que la API pueda ser consumida localmente, y una vez hecho lo anterior, es nevesario teclear desde terminal de `VSCode` o
desde la terminal donde accediste a tu proyecto. El siguiente comando: **uvicorn main:app --reload** `enter` para obtener  en tu localhost:http://127.0.0.1:8000/docs.
* Una vez, probada la API localmente puedes continuar en `VSCode` para agregar el archivo txt: `requirements.txt` creado vacío y agregando solamente las librerias que se crean convenientes. 
* Enseguida subí mi proyecto a GitHub, en mi caso y desde terminal de VSCode, escribí los comandos: **git add .**`enter`, **git commit -m 'PI_ML_OPS'** `enter` y **git push** `enter`.
* Después accedí a mi cuenta de GitHub, abriendo mis repositorios para ubicar mi proyecto: **PI_ML_OPS** para abrirlo e inspeccionar que todo estaba bien.
* Finalmente entramos a: https://render.com, y conectamos con la cuenta GitHub para acceder a un `WebService` para cargar el proyecto, para que la `API` este disponible y consumible en la `Nube`, que si todo esta bien: 
 el icono de color verde `live` aparece y en todos los casos aparentemente se puede ver el link de tu API. En mi caso: https://fastapi-cb8b.onrender.com
                                         
## **Sistema de recomendación**
Una vez creados los insumos de las funciones de recomendación, las cuáles son representadas por: def <b>recomendacion_juego(<em>titulo</em>)</b> y def <b>recomendacion_usuario(<em>user_id</em>)</b>. Creamos las respectivas funciones de recomendación basadas en los siguientes datasets: <i>recomendacion_juego.csv</i> y <i>recomendacion_usuario.csv</i> para integrarlas en la lista de funciones creadas en el
notebook: <em>Funciones_API</em>

<h3>Acceso Repositorio GitHub</h3>

https://github.com/JBE777/PI_ML_OPS

<h3>Acceso Deploy Render</h3>

https://fastapi-cb8b.onrender.com/docs#/
                                                                                                                     
<h3>Acceso Video</h3>
<p><img src="formas/Git.png", width="150"></p>
<h6 align=right>México - 2023</h6>