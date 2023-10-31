import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fastapi import FastAPI

app = FastAPI()

@app.get("/PlayTimeGenre/{genero}")
def PlayTimeGenre(genero:str):
    '''Debe devolver año con más horas jugadas para dicho género.Ejemplo de retorno: 
    {"Año de lanzamiento con más horas jugadas para el Género X" : 2013}
    con Input:genero'''
    df = pd.read_csv('./genero.csv')
    d = df.loc[df.genero==genero]
    e = d.playtime_hours.count()
    f = d.release_year.to_list()
    return {'Año de lanzamiento':f[0]}

@app.get("/UserForGenre/{genero}")
def UserForGenre(genero:str): 
    '''Debe devolver el usuario que acumula más horas jugadas para el género dado y una 
    lista de la acumulación de las horas jugadas por año. Ejemplo de retorno: 
    {"Usuario con más horas jugadas para el Género X" : us213ndjss09sdf, "Horas jugadas":
    [{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]}
    con Input:genero'''
    df = pd.read_csv('./userforgenre.csv')
    gen = df.loc[df.genero==genero]
    u = gen.user_id.to_list()[0]
    d = gen.posted_year.to_list()
    f = gen['playtime_hours'].to_list()
    return {'Usuario por género':u,'Horas jugadas':[{'Año':d[0], 'Horas':int(round(f[0],0))},
           {'Año':d[1], 'Horas':int(round(f[1],0))},{'Año':d[2], 'Horas':int(round(f[2],0))}]}
    
@app.get("/UsersRecommend/{anio}")
def UsersRecommend(anio:int): 
    '''Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = 
    True y comentarios positivos/neutrales). Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},
    {"Puesto 3" : Z}] con Input:anio=Año'''
    df = pd.read_csv('./UsersRecommend.csv')
    a = df.loc[df.posted_year==anio]
    b = a.sentiment_analysis.count()
    c = a.title.to_list()
    return {'Puesto 1':c[0],'Puesto 2':c[1],'Puesto 3':c[2]}

@app.get("/UsersNotRecommend/{anio}")    
def UsersNotRecommend(anio:int): 
    '''Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = 
    False y comentarios negativos). Ejemplo de retorno: [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}]
    con Input:anio=Año'''
    df = pd.read_csv('./UsersNotRecommend.csv')
    d = df.loc[df.posted_year==anio]
    e = d.sentiment_analysis.count()
    f = d.title.to_list()
    return {'Puesto 1':f[0],'Puesto 2':f[1],'Puesto 3':f[2]}

@app.get("/sentiment_analysis/{anio}")    
def sentiment_analysis(anio:int): 
    '''Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas
       de usuarios que se encuentren categorizados con un análisis de sentimiento. Con Input:anio=Año'''
    df = pd.read_csv('./sentimientos.csv')
    d = df[df['release_year']==anio]
    e = d['review'].count()
    f = d[d['sentiment_analysis']==0.0].count()
    g = d[d['sentiment_analysis']==1.0].count()
    h = d[d['sentiment_analysis']==2.0].count()
    neg = (f/e)*len(d)
    neu = (g/e)*len(d)
    pos = (h/e)*len(d)
    a = neg.to_list()
    b = neu.to_list()
    c = pos.to_list()
    return {'Negative':int(a[0]), 'Neutral':int(b[0]),'Positive':int(c[0])}


@app.get("/recomendacion_juego/{id_producto}")
def recomendacion_juego(id_producto):
    '''Ingresando el 'id' de un producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.'''
    df = pd.read_csv('./recomendacion_juego.csv')
    f = df[df.id==id_producto]
    lista = f.title.to_list()
    return {'Recomendacion_juego':lista[:5]}

@app.get("/recomendacion_usuario/{user_id}")
def recomendacion_usuario(user_id):
    '''Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.'''
    g = pd.read_csv('./recomendacion_usuario.csv')
    tfidf = TfidfVectorizer(stop_words = 'english')
    g['user_id'] = g['user_id'].fillna('')
    tfidf_matriz = tfidf.fit_transform(g['user_id']) 
    coseno_sim = linear_kernel(tfidf_matriz,tfidf_matriz)
    indices = pd.Series(g.index, index = g['user_id'])
    idx = indices[user_id]
    simil = list(enumerate(coseno_sim[idx]))
    simil = sorted(simil, key = lambda x: x[0],reverse=True)
    simil = simil[1:11]
    usuario_index = [g[0] for g in simil]
    lista = g['title'].iloc[usuario_index].to_list()[:5]
    return {'Recomendacion_usuario':lista}