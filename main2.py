import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
from scipy import stats
from scipy.stats import *
import matplotlib.pyplot as plt
from numpy import float_, int_, ndarray
from scipy.stats import gaussian_kde
import base64
from io import BytesIO
import scipy.special as sc
import xlsxwriter
import statsmodels.api as sm
import math as mt




def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', float_format="%.2f")
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="ASI.xlsx"> Download do Resultado </a>' # decode b'abc' => abc


def boxplot(data_frame):


    tam = len(data_frame.columns)


    #fig, ax = plt.subplots(tam,2,figsize=(15,15 + 2*tam))

    #line = st.slider("Largura", min_value=4, max_value=40)
    #col = st.slider("Altura", min_value=5.6, max_value=40)
    fig, ax = plt.subplots(tam,2, figsize=(5.6*2,4.6*tam))

    descricao = pd.DataFrame()

    p=0
    for col in data_frame.columns:

        for i in range(len(data_frame.index)):
            data_frame.loc[i, col] = str(data_frame.loc[i, col]).replace(',', '.')

        x = (data_frame[col]).astype(np.float64).dropna()
        descricao[p] =x.describe()
        n = x.count()
        dt = x

        if(n<4):
            ax[p,0].hist(dt, density=True, histtype='step', color="#2466b6", alpha=.9, label='Histograma')
            ax[p,0].legend()
            ax[p, 0].title.set_text(col)
        else:
            KDEpdf = gaussian_kde(dt, bw_method='silverman')
            x = np.linspace((0.8*dt.min()), (dt.max()*1.2), 2000)
            #print(x)
            ax[p,0].plot(x, KDEpdf(x), 'r', color="black", linestyle='-', lw=3, label='Densidade aproximada')
            ax[p,0].hist(dt, density=True, histtype='step', color="#2466b6", alpha=.9, label='Histograma')
            ax[p,0].title.set_text(col)

            ax[p,0].legend()

        green_diamond = dict(markerfacecolor='g', marker='D')
        ax[p,1].boxplot(dt,notch=False, showfliers=True, flierprops=green_diamond)
        ax[p, 1].title.set_text(col)



        p = p + 1
        #plt.show()
    descricao.columns = data_frame.columns
    descricao.index = ['itens','média','desvio padrão','valor mínimo','25%','mediana','75%','valor máximo']
    st.table(descricao)

    st.pyplot(fig)
    plt.close()




    return

    #plt.close()


def plotar(serie,item):
    #tam = 1
    fig, ax = plt.subplots(2, figsize=(5.6, 4.6*2))

    x = serie.astype(np.float64).dropna()

    dt = x


    KDEpdf = gaussian_kde(dt, bw_method='silverman')
    x = np.linspace((0.8 * dt.min()), (dt.max() * 1.2), 2000)
    # print(x)
    ax[0].plot(x, KDEpdf(x), 'r', color="black", linestyle='-', lw=3, label='Densidade aproximada')
    ax[0].hist(dt, density=True, histtype='step', color="#2466b6", alpha=.9, label='Histograma')
    ax[0].title.set_text(item)

    ax[0].legend()

    green_diamond = dict(markerfacecolor='g', marker='D')
    ax[1].boxplot(dt, notch=False, showfliers=True, flierprops=green_diamond)
    ax[1].title.set_text(item)
    st.pyplot(fig)
    plt.close()


def plotar2(serie1, serie2, item):

    fig, ax = plt.subplots(1, figsize=(5.6, 4.6*2*1))

    y = serie1.astype(np.float64).dropna()
    x = serie2.astype(np.float64).dropna()


    ax.title.set_text(item)

    b1, bo, r_value, p_value, std_err = stats.linregress(x, y)


    yhat = bo + b1 * x


    # estimate stdev of yhat
    sum_errs = ((y - yhat) ** 2).sum()
    stdev = mt.sqrt(1 / (len(y) - 2) * sum_errs)
    # calculate prediction interval
    interval = 1.96 * stdev
    #print('Prediction Interval: %.3f' % interval)




    ax.scatter(x, y)

    plt.plot(x,yhat,color='red')


    ax.set_xlabel("R²: " + str(round(r_value * r_value,2)), size=15)

    st.pyplot(fig)
    plt.close()

    #st.write(y)
    #st.write(y)

    return r_value * r_value



def boxplot2(data_frame):


    df_filter = data_frame

    for col in data_frame.columns:

        data_frame[col] = data_frame[col].astype(str)
        data_frame[col] = data_frame[col].str.replace(',', '.')
        data_frame[col] = data_frame[col].astype(float)
        data_frame[col].dropna(inplace=True)


    df = data_frame
    df = df.loc[:, ::2]

    col1, col2, col3 = st.beta_columns(3)

    with col1:
        #st.header("Escolha o item")
        item = st.radio("Escolha",(df.columns))

        pos = data_frame.columns.get_loc(str(item))

        x = data_frame.iloc[:,pos+1]
        x1 = df_filter.iloc[:, pos + 1]
        min_global = np.float(x.min())
        max_global = np.float(x.max())
        min_local = np.float(x1.min())
        max_local = np.float(x1.max())

        a,b = st.slider("Filtrar quantidade", min_global, max_global, (min_local, max_local), 1.0)

        nome_coluna = data_frame.columns[pos+1]

        logic = (((data_frame[nome_coluna]).astype(np.float64) >= a) & ((data_frame[nome_coluna]).astype(np.float64) <= b))

        if(logic.sum() < 3):
            st.markdown("""***É preciso selecionar ao menos 3 amostras***""")

        else:
            df_filter = data_frame[logic]



        tabela_check = st.checkbox("Mostrar tabela")
        if(tabela_check):
            st.table(df_filter[df_filter.columns[pos:pos+2]])
            #st.table(df_filter.iloc[:,pos:pos + 2])




    with col2:
        #st.header("Veja os graficos")
        #st.write(data_frame[str(item)])
        plotar(df_filter[str(item)],item)



    with col3:

        pos = df_filter.columns.get_loc(str(item))
        #st.write(data_frame.iloc[:,[pos,pos+1]])
        serie1 = df_filter.iloc[:,pos]
        serie2 = df_filter.iloc[:,pos+1]



        #st.write(serie2)
        r2 = plotar2(serie1, serie2, item)







    return df_filter, item, a, b, r2

from sklearn.metrics import mean_squared_error

def icRegressao(df_filter,item,nc, xh):


    # https://online.stat.psu.edu/stat501/lesson/3/3.3
    pos = df_filter.columns.get_loc(str(item))
    #st.write(df_filter.iloc[:,[0,0+1]])



    y = df_filter.iloc[:, 0].astype(np.float64).dropna()
    x = df_filter.iloc[:, 0 + 1].astype(np.float64).dropna()

    n_1 = df_filter[item].count()

    b1, bo, r_value, p_value, std_err = stats.linregress(x, y)




    yhat = bo + b1 * x



    # estimate stdev of yhat
    sum_errs = ((y - yhat) ** 2).sum() / (len(y) - 2)




    sigma2_est = sum_errs/ (len(y) - 2) #MSE

    t3 = (np.float(xh) - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()

    erroPI = mt.sqrt(sigma2_est * (1 + 1 / len(y) + t3))

    # calculate prediction interval
    t_coeficiente = t.ppf(0.5 + 0.5 * nc, n_1 - 2)
    interval = erroPI*t_coeficiente

    #print('Prediction Interval: %.3f' % interval)

    yhat_out = bo + b1 * np.float(xh)


    lower, upper = yhat_out - interval, yhat_out + interval


    return round(yhat_out,2), round(lower,2), round(upper,2)


def dp_MJ(data, p):
    data = np.sort(data)  # conjunto ordenado
    n = data.size  # n

    betacdf = beta.cdf

    x = np.arange(1, n + 1, dtype=float_) / n
    y = x - 1. / n

    w = sc.betainc(p * (n + 1), (n + 1) * (1 - p), x) - sc.betainc(p * (n + 1), (n + 1) * (1 - p), y)
    # print(w)

    # W = betacdf(x,p*(n+1),(n+1)*(1-p)) - betacdf(y,p*(n+1),(n+1)*(1-p))
    # print(W)

    C1 = np.dot(w, data)  # valor esperado E[X]
    # print(C1)


    C2 = np.dot(w, data ** 2)  # valor esperado E[X^2]

    mj = np.sqrt(C2 - C1 ** 2)

    return mj


def medianaIgor(data, p):
    data = np.sort(data)  # conjunto ordenado
    n = data.size  # n

    betacdf = beta.cdf

    x = np.arange(1, n + 1, dtype=float_) / n
    y = x - 1. / n

    w = sc.betainc(p * (n + 1), (n + 1) * (1 - p), x) - sc.betainc(p * (n + 1), (n + 1) * (1 - p), y)
    # print(w)

    # W = betacdf(x,p*(n+1),(n+1)*(1-p)) - betacdf(y,p*(n+1),(n+1)*(1-p))
    # print(W)

    C1 = np.dot(w, data)  # valor esperado E[X]
    # print(C1)
    C2 = np.dot(w, data ** 2)  # valor esperado E[X^2]

    mj = np.sqrt(C2 - C1 ** 2)

    return C1



def abrirArquivo(arquivo):
    df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8', low_memory=False)
    return df

def processaDf(df, mr=0, nc=0.95):
    obs = []
    n_out = []
    ls = []
    li = []
    mediana = []


    for col in df.columns:

        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(',', '.')
        df[col] = df[col].astype(float)
        df[col].dropna(inplace=True)

        n_0 = df[col].count()
        df[col] = remover_outliers(df[col], mr)
        n_1 = df[col].count()

        n_out.append(n_0-n_1)


        if (n_1 < 3):
            obs.append("Menos de 3 amostras válidas")
            mediana.append(0)
            ls.append(0)
            li.append(0)
        else:
            obs.append("Mais de 3 amostras válidas")

            t_coeficiente = t.ppf(0.5 + 0.5 * nc, n_1 - 1)


            dados = df[col].dropna()

            mediana_aux = medianaIgor(dados,0.5)
            dp_aux = dp_MJ(dados,0.5)

            mediana.append(mediana_aux)
            ls.append(mediana_aux + t_coeficiente*dp_aux/2)
            li.append(mediana_aux - t_coeficiente*dp_aux/2)



        x = pd.DataFrame()
        #x['Média'] = df.mean().astype(float)
        x['Outliers Removidos'] = n_out
        x['Observação'] = obs
        x['Mediana'] = mediana
        x['Limite Inferior'] = li
        x['Limite Superior'] = ls

    return x



def processaDf2(df, col, mr=0, nc=0.95,a=0,b=0, r2=0, qtd=0, estimador="Mediana - Distribuição Livre"):

    obs = []
    n_out = []
    n = []
    ls = []
    li = []
    est = []



    n_0 = df[col].count()
    df[col] = remover_outliers(df[col], mr)
    n_1 = df[col].count()

    logic = pd.isna(df)

    df = (df[~logic[col]])

    if mr == 0:
        obs_rm = "Optou-se por não utilizar método de remoção de outliers." + "  "
    if mr == 1:
        obs_rm = "A remoção de Outliers foi realizada pelo método Boxplot. " + " "
    if mr == 2:
        obs_rm = "A remoção de Outliers foi realizada pelo método z-score." + " "



    n_out.append(n_0 - n_1)

    if (n_1 < 3):
        obs.append("Menos de 3 amostras válidas")
        est.append(0)
        ls.append(0)
        li.append(0)
    else:


        t_coeficiente = t.ppf(0.5 + 0.5 * nc, n_1 - 1)

        dados = df[col].dropna()

        if(estimador == "Mediana - Distribuição Livre"):
            mediana_aux = medianaIgor(dados, 0.5)
            dp_aux = dp_MJ(dados, 0.5)

            est.append(mediana_aux)
            ls.append(mediana_aux + t_coeficiente * dp_aux / 2)
            li.append(mediana_aux - t_coeficiente * dp_aux / 2)

            obs.append(" A etapa de investigação dos dados filtrou as quantidades dos elementos da amostra para intervalo de " + str(a) + " - " + str(b) +
                       ". " + obs_rm + "Foi utilizado o estimador não paramétrico de distribuição livre para a mediana com "
                       + "nível de confiança de " + str(round(nc*100)) + "%"
                       +  ". O valor do coeficiente de determinação vale R²=" + str(round(r2, 2)))

        if (estimador =="Mínimos Quadrados"):

            y_out, lower, upper = icRegressao(df,item,nc,qtd)

            est.append(y_out)
            ls.append(upper)
            li.append(lower)
            obs.append(
                " A etapa de investigação dos dados filtrou as quantidades dos elementos da amostra para intervalo de " + str(round(a)) + " até " + str(round(b)) + ". "
                + obs_rm + "Foi utilizada a predição pelo método dos mínimos quadrados com "
                + "nível de confiança de " + str(round(nc * 100)) + "% " + "para uma quantidade questionada igual a " + str(qtd)
                + ". O valor do coeficiente de determinação vale R²=" + str(round(r2, 2)))



    n.append(n_1)

    x = pd.DataFrame()





    x['amostra'] =  n

    x['Outliers Removidos'] = n_out

    if (estimador == "Mediana - Distribuição Livre"):
        x['Estimador Mediana'] = est

    if (estimador == "Mínimos Quadrados"):
        x['Estimador MQ'] = est

    x['Limite Inferior'] = li
    x['Limite Superior'] = ls
    x['Observação'] = obs


    return x



def remover_outliers(serie = pd.Series([],dtype="float64"), mr=0):

    iq = serie.quantile(0.75) - serie.quantile(.25)
    ls = serie.quantile(0.75) + 1.5*iq
    lin = serie.quantile(0.25) - 1.5*iq

    if(mr == 2):
        z = np.abs(stats.zscore(serie))
        for i,v in serie.items():
            if (z[i] > 3):
                serie[i] = np.nan

    if (mr == 1):
        serie[ (serie > ls ) | (serie < lin ) ] = np.nan

    return serie




def _max_width_():
    max_width_str = f"max-width: 1200px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

st.write("""
# Acubens - ASI
  *Ferramenta para auxiliar na análise de indicativo estatístico de sobrepreço*
 """)

_max_width_()


#video_file = open('C:/Users/dpf.adm/Desktop/Streamlite-Teste/tutorial.mp4', 'rb')
#video_bytes = video_file.read()



with st.beta_expander("Executar Análise Estatística"):
    st.subheader("""  **Etapa 1  - Leitura do arquivo** """)

    uploaded_file = st.file_uploader("Escolha um arquivo", ["csv"])


    if uploaded_file is not None:


        df = abrirArquivo(uploaded_file)

        st.write(df)

        st.subheader("""  **Etapa 2 - Investigação dos Dados: Distribuição, Histograma e Boxplot** """)


        df, item, a,b, r2 = boxplot2(df)

        st.markdown(""" **Resumo Estatístico Descritivo**""")
        descricao = df.describe()
        descricao.index = ['Tamanho da amostra', 'média', 'desvio padrão', 'valor mínimo', '25%', 'mediana', '75%',
                           'valor máximo']

        pos = df.columns.get_loc(str(item))
        st.table(descricao.iloc[:,pos:(pos+2)])

        left_column, center_column, right_column = st.beta_columns(3)
        # You can use a column just like st.sidebar:
        with left_column:
            st.subheader("""  **Etapa 3 - Remoção de Outliers** """)
            mr = st.radio('Método',("Nenhum", "BoxPlot", "z-score"))

            if mr == "Nenhum":
                mr = 0

            if mr == "BoxPlot":
                mr = 1

            if mr == "z-score":
                mr = 2

        with center_column:
            st.subheader("""  **Etapa 4 - Escolha do Estimador** """)
            estimador = st.radio('Estimador', ("Mediana - Distribuição Livre", "Mínimos Quadrados"))

            if estimador =="Mediana - Distribuição Livre":
                #st.write("")
                qtd = 0
            if estimador =="Mínimos Quadrados":
                #st.write("")
                qtd = st.text_input("Digite a quantidade",0)
                #icRegressao(df, item, nc, np.float(qtd))


        with right_column:
            st.subheader("""  **Etapa 5 - Intervalo de Confiança** """)
            nc = st.radio('Nível de Confiança', ("95%", "99%", "90%"))

            if nc == "95%":
                nc = 0.95

            if nc == "99%":
                nc = 0.99

            if nc == "90%":
                nc = 0.90



        pos = df.columns.get_loc(str(item))
        if ((pos +2) >= len(df.columns)):
            x = processaDf2(df.iloc[:, pos::], item, mr, nc, a, b, r2, qtd, estimador)
            #st.write(df.iloc[:, pos::])
        else:
            x= processaDf2(df.iloc[:,pos:(pos+2)], item, mr, nc, a ,b, r2, qtd, estimador)
            #st.write(df.iloc[:, pos:(pos+2):])



        #st.markdown("***Resumo da análise para o item:*** " + str(df.columns[pos]))


        y = pd.DataFrame()


        y['Amostra analisada (n)'] = x['amostra']
        y['Outliers Removidos'] = x['Outliers Removidos']

        if (estimador == "Mediana - Distribuição Livre"):
            y['Estimador Mediana'] = x['Estimador Mediana']
        if (estimador == "Mínimos Quadrados"):
            y['Estimador MQ'] = x['Estimador MQ']

        y['Limite Inferior'] = x['Limite Inferior']
        y['Limite Superior'] = x['Limite Superior']

        y['Observação'] = x['Observação']

        y.index = [str(df.columns[pos])]

        st.table(y)

        #plotly_table(y)
        #st.table(df[df.columns[pos:pos+2]])
        #st.table(df)



        #df2 = y  # your dataframe

        #st.write("Cique no botão para gerar um arquivo compatível com Excel")



        #if st.button("Gerar arquivo"):
            #st.markdown(get_table_download_link(df2), unsafe_allow_html=True)









with st.beta_expander("Tutorial"):
    st.write(""" 
    *Em construção*
    
    **Caso seja sua primeira vez utilizando o Acubens - ASI, confira o tutorial em  abaixo**
    """)

    #st.video(video_bytes)



with st.beta_expander("Metodologia e Métodos Matemáticos "):

    st.header("Metodologia")
    st.write(r"""
    São disponibilizadas ferramentas técnicas de análise de dados para auxiliar o usuário em sua análise
    A aplicação consiste em 3 etapas a serem percorridas, conforme descrito abaixo.
       """)
    st.subheader("1. Investigação dos Dados:")
    st.write(r"""

    1.1  Histograma, Distribuição estimada, Boxplot para a variável ***Preço***
    
    1.2  Gráfico de Dispersão: ***Quantidade x Preço***
    
    1.3  Seleção de subconjunto através da filtragem dos limites para o intervalo da variável ***Quantidade***


    """)


    st.subheader("2. Métodos de estimação:")
    st.write(r"""
    
    2.1 Estimador não paramétrico de distribuição livre para a **Mediana** populacional
    
    2.2 Estimador Mínimos Quadrados 
    
    """)

    st.subheader("3. Síntese da Análise")
    st.write(r"""

    Apresentação de tabela com síntese da análise realizado e disponibilização de texto resumo gerado dinamicamente. 

    """)


    st.header("Métodos Matemáticos")

    st.write("""
    Esta seção é destinada a apresentação da formulação matemática para os métodos de estimação disponíveis nesta aplicação
    
    """)
    st.subheader("Estimador não paramétrico de distribuição livre para a **Mediana** populacional")

    st.write(r"""
    Seja $X_{1}\text{,...,}X_{n}$, uma amostra aleatória de tamanho  $n$  retirada de uma distribuição contínua com função de distribuição  $F(.)$. 
    Considere  $X_{(1)}\text{,...,} X_{(n)}$  a estatística ordenada da amostra e o vetor  $X =(X_{(1)}\text{,...,} X_{(n)})$.

    O valor esperado da k-ésima estatística ordenada é dado por:
    $$
    E(X_{(k)}) = { 1 \over \beta(k,n-k+1)} \displaystyle\int_{-\infty}^{+\infty}xF(x)^{k-1}  (1 - F(x))^{n-k}dF(x)
    $$
    
    $$
    = { 1 \over \beta(k,n-k+1)} \displaystyle\int_{0}^{1}F^{-1}(y)y^{k-1}(1-y)^{n-k}dy
    $$
    
    Dado que $E(X_{(n+1)p})$ converge para $F^{-1}(p)$ para $p\in(0,1)$, toma-se como estimador para $F^{-1}(p)$ o valor esperado para $E(X_{(n+1)p})$, sendo $(n+1)p$ inteiro ou não:
    
    $$
    Q_p = { 1 \over \beta((n+1)p,(n+1)(1-p))} \displaystyle\int_{0}^{1}F_n^{-1}(y)y^{(n+1)p-1}(1-y)^{(n+1)(1-p)}dy
    $$
    
    Onde $F_n(X)$ é a função de distribuição acumulada, $F_{n}(X) = n^{-1}\sum I(X_i \leq x)$ ,  $I(A)$ é a função indicadora do conjunto $A$. O estimador pode ser reescrito como:
    
    $$
    Q_p = \displaystyle\sum_{i=1}^{n} W_{n,i} X_{(i)}
    $$
    
    Onde,
    
    $$
    W_{n,i} = { 1 \over \beta((n+1)p,(n+1)(1-p))} \displaystyle\int_{(i-1)/n}^{i/n}y^{(n+1)p-1}(1-y)^{(n+1)(1-p)}dy
    
    $$
    
    $$
    = I_{i/n}(p(n+1),(1-p)(n+1)) - I_{(i-1)/n}(p(n+1),(1-p)(n+1))
    $$
    
    e $I_{x}(a,b)$ denota a função incompleta de beta
    
    Generalizando o procedimento descrito acima para o cálculo do momento de ordem $l$, temos o estimador genérico : 
   
    $$
    C_d = \displaystyle\sum_{i=1}^{n} W_{n,i} X_{(i)}^d
    $$
    
    Conforme proposto em [1], utilizando os momentos de primeira e segunda ordem podemos expressar o desvio padrão como:
    $$
    S_p = \sqrt{C_2 - C_1^2}
    $$
    
    Fazendo $p=0.5$, temos $Q_{0.5}$ como o estimador para a mediana e $S_{0.5}$ como desvio padrão associado.


    Define-se o intervalo de confiança com nível de confiança $100(1-\alpha)\%$:
    
    $$
    \boxed{IC_{\alpha} = Q_{0.5} \pm t_{\alpha,n-1} S_{0.5}}
    $$
    
    onde $\alpha$ é o nível de significância e $t_{\alpha,n-1}$ é o valor da distribuição bicaudal **t-Student** para $n-1$ graus de liberdade.  
    
       

    """)

    st.subheader("Estimador Mínimos Quadrados ")

    st.write(r"""
    
    Para a utilização deste método, os seguintes requisitos devem ser observados:
    
    1) A Quantidade questionada deve estar dentro do escopo do modelo, ou seja, dentro dos limites utilizados para a modelagem
    
    2) O erro deve ser independente, possuir distribuição normal e ser homocedástico. O requisito da normalidade recai sobre a necessidade de se conhecer a distribuição dos erros para o cálculo do intervalo de confiança.
    
    
    São utilizados os pares $(x_{i},y_{i})$, onde $x_{i}$ é a quantidade observada e $y_{i}$ é o preço observado. Pode-se estabelecer uma regressão linear simples cujo modelo estatístico é:
    

    $$
    {Y}_{i} = b_{0} + b_{1}x_{i} + \epsilon_{i}
    $$
    
    $Y_{i}$ é uma variável aleatória e representa o valor da variável resposta (variável dependente) - **Preço** 
    
    $x_{i}$ representa o valor da variável explicativa (variável independente) - **Quantidade**
    
    $b_{0}$ e $b_{1}$ são os parâmetros do modelo, que serão estimados, e que definem a reta de regressão
    
    As estimativas de mínimos quadrados para os parâmetros $b_{0}$ e $b_{1}$ são:
    
    $$
    \hat{b}_{1} = \frac{S_{xy}}{S_{xx}} 
    $$
    
    $$
    \hat{b}_{0} = \bar{Y} - \hat{b}_{1} \bar{x}
    $$
    
    A Soma de Quadrados dos Resíduos (erros),
    $$
    SQE = \displaystyle\sum_{i=1}^{n} \epsilon_{i}^{2}
    $$
    É um estimador viciado para variância residual (dos erros) $\sigma^{2}$, pois $E[SQE] = (n-2) \sigma^{2}$.
    
    Desta forma, utiliza-se o estimador não viciado dado por
    $$
    \hat{\sigma}^{2} = QME = \frac {SQE}{(n-2)}
    $$
    $QME$ é o Quadrado Médio dos Erros.
    
    Com os parâmetros do modelo definido, pode-se estimar uma resposta a partir de um valor observado $x_{h}$, tal que $\hat{y}_{h} = \hat{b}_{0} + \hat{b}_{1}x_{h}$.
    
    
    O intervalo de confiança para a predição $\hat{y}_{h}$ é definido como:
    
    $$
    \boxed{IC_{\alpha} = \hat{y}_{h} \pm t_{\alpha,n-2} \sqrt{QME \left( 1 + \frac 1 n +  \frac {x_{h} - \bar{x}} {\displaystyle\sum_{i=1}^{n}(x_{i} - \bar{x})}  \right) }}
    $$
    
    

       """)

    #simu = pd.read_csv("C:/Users/dpf.adm/Desktop/Streamlite-Teste/simu.csv", sep=';', encoding='utf-8', low_memory=False)
    #simu_tabela = st.table(simu)


with st.beta_expander("Sobre", expanded=True):
    st.write(""" 
    Idealizada pelos Peritos Criminais Federais: **Igor Gameleiro**, **Rafaela da Fonte** e **Vitor Gomes**, a ferramenta busca auxiliar na análise de sobrepreço a partir de técnicas de visualização de dados e de inferência estatística. 
    
    **Atenção:** Esta aplicação **não** é normatizada pela **Policia Federal**. O usuário tem total responsabildiade sobre o uso da mesma. O código fonte pode ser obtido através de solicitação ao email: gameleiro.iog@pf.gov.br
    
    """)


