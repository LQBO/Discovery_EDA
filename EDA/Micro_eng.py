import cv2 as cv
import numpy as np
import pandas as pd
import os


def Microplate(am, title):


    def click(event, x, y, flag, param):
        ix = x
        iy = y
        if event == cv.EVENT_LBUTTONDOWN:
            for i in circles[0, :]:
                menorX = (i[0] - i[2])
                maiorX = i[0] + i[2]
                menorY = (i[1] - i[2])
                maiorY = i[1] + i[2]
                if i[0] < i[2]:
                    menorX = 0
                if i[1] < i[2]:
                    menorY = 0
                if ix >= menorX and ix <= maiorX and iy >= menorY and iy <= maiorY:
                    center = (i[0], i[1])
                    radius = i[2]
                    raios.append(radius)
                    if len(S[h]) < am:
                        if S[h].count(center) < 1:
                            cv.circle(image, center, radius, (0, 0, 255), 3)
                            S[h].append(center)
                            font = cv.FONT_HERSHEY_SIMPLEX
                            cv.putText(image, f'ID{id[h]}', (ix, iy), font, .5, (0, 0, 0), 2)
                            cv.imshow('image', image)
                        for g in range(1, am+1):
                            if g != h:
                                if center in S[g]:
                                    cv.circle(image, center, radius, (255, 0, 0), 3)  # círculo azul: contagem 2
                                    cv.imshow('image', image)


    S = {}
    testes = []
    padrao = {}
    pxamostra_b = {}
    pxamostra_g = {}
    pxamostra_r = {}
    raios = []
    pxpadrao_b = {}
    pxpadrao_g = {}
    pxpadrao_r = {}
    id = {}
    ID1 = []
    ID2 = []
    IDpx = []
    cutoff = 5
    df = pd.DataFrame(columns=['ID1', 'ID2', 'pixel'])

    dados = pd.read_csv('pxeda.csv')
    df2 = pd.DataFrame(dados)
    df3 = {}

    for h in range(1, am+1):
        S[h] = []
        id[h] = int(input(f'for sample {h}, what is the sample ID: '))

    raiz_path = os.path.dirname(os.path.abspath(__file__))
    pasta = str('/Micro/')

    # importando imagem
    image = cv.imread(raiz_path + pasta + title)
    scale_percent = 25  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

    print('Resized Dimensions : ', image.shape)
    cv.imshow("Resized image", image)
    cv.waitKey(0)
    image2 = image.copy()
    image3 = image.copy()


    # separando os filtos
    b, g, r = cv.split(image3)
    '''cv.imwrite('blue.jpg', b)
    cv.imwrite('green.jpg', g)
    cv.imwrite('red.jpg', r)'''
    gray_inv_b = cv.bitwise_not(b)
    gray_inv_g = cv.bitwise_not(g)
    gray_inv_r = cv.bitwise_not(r)

    '''cv.imwrite('blue_inv.jpg', gray_inv_b)
    cv.imwrite('green_inv.jpg', gray_inv_g)
    cv.imwrite('red_inv.jpg', gray_inv_r)'''

    # gray scale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_inv = cv.bitwise_not(gray)


    #reconhecendo circulos

    gray_inv = cv.medianBlur(gray_inv, 5)
    rows = gray_inv.shape[0]
    circles = cv.HoughCircles(gray_inv, cv.HOUGH_GRADIENT, 1, rows / 8, param1=15, param2=9, minRadius=30, maxRadius=60)

    if circles is not None:
        circles = np.uint64(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(image, center, radius, (255, 0, 255), 3)


    '''cv.imshow("detected circles", image)
    cv.waitKey(0)'''
    cv.imshow('image', image)
    for h in range(1, am+1):
        cv.setMouseCallback('image', click)
        cv.waitKey(0)

    for c in range(1, am+1):
        padrao[c] = S[c].copy()

    for w in range(1, am+1):
        for centro1 in padrao[w]:
            for c in range(1, am + 1):
                if w + c <= am:
                    for centro2 in padrao[w+c]:
                        if centro1 == centro2:
                            testes.append(centro1)

    for items in testes:
        for indices, centros in padrao.items():
            if items in centros:
                padrao[indices].remove(items)

    #Contando o número de pixel no padrão
    met_raio = np.mean(raios)/2

    for i in range(1, am+1):
        janela_b = gray_inv_b[int(padrao[i][0][1]-met_raio):int(padrao[i][0][1]+met_raio), int(padrao[i][0][0]-met_raio):int(padrao[i][0][0]+met_raio)]
        pxpadrao_b[i] = int(np.mean(janela_b))
        janela_g = gray_inv_g[int(padrao[i][0][1] - met_raio):int(padrao[i][0][1] + met_raio), int(padrao[i][0][0] - met_raio):int(padrao[i][0][0] + met_raio)]
        pxpadrao_g[i] = int(np.mean(janela_g))
        janela_r = gray_inv_r[int(padrao[i][0][1] - met_raio):int(padrao[i][0][1] + met_raio), int(padrao[i][0][0] - met_raio):int(padrao[i][0][0] + met_raio)]
        pxpadrao_r[i] = int(np.mean(janela_r))


    #Contando o px nas amostras:
    for w in range(1, am + 1):
        cv.putText(image2, f'ID-{id[w]}', (int(padrao[w][0][0] - 10), int(padrao[w][0][1] - 2)), cv.FONT_HERSHEY_SIMPLEX, .5,(0, 0, 0), 2)
        for centro1 in S[w]:
            for c in range(1, am + 1):
                if w + c <= am:
                    for centro2 in S[w + c]:
                        if centro1 == centro2:
                            pxamostra_b[w, w + c] = int(np.mean(gray_inv_b[int(centro1[1]-met_raio):int(centro1[1]+met_raio), int(centro1[0]-met_raio):int(centro1[0]+met_raio)]))
                            pxamostra_g[w, w + c] = int(np.mean(gray_inv_g[int(centro1[1] - met_raio):int(centro1[1] + met_raio), int(centro1[0] - met_raio):int(centro1[0] + met_raio)]))
                            pxamostra_r[w, w + c] = int(np.mean(gray_inv_r[int(centro1[1] - met_raio):int(centro1[1] + met_raio), int(centro1[0] - met_raio):int(centro1[0] + met_raio)]))
                            if pxpadrao_b[w] >= pxpadrao_b[w+c]:
                                pxeda_b = pxamostra_b[w, w+c] - pxpadrao_b[w]
                            else:
                                pxeda_b = pxamostra_b[w, w+c] - pxpadrao_b[w+c]

                            if pxpadrao_g[w] >= pxpadrao_g[w + c]:
                                pxeda_g = pxamostra_g[w, w + c] - pxpadrao_g[w]
                            else:
                                pxeda_g = pxamostra_g[w, w + c] - pxpadrao_g[w + c]

                            if pxpadrao_r[w] >= pxpadrao_r[w + c]:
                                pxeda_r = pxamostra_r[w, w + c] - pxpadrao_r[w]
                            else:
                                pxeda_r = pxamostra_r[w, w + c] - pxpadrao_r[w + c]
                            if pxeda_b <= cutoff:
                                pxeda_b = 0
                            if pxeda_g <= cutoff:
                                pxeda_g = 0
                            if pxeda_r <= cutoff:
                                pxeda_r = 0
                            pxeda = pxeda_b + pxeda_g + pxeda_r
                            if pxeda == 0:
                                cv.circle(image2, centro1, int(met_raio*1.5), (0, 0, 255), -1)
                                ID1.append(id[w])
                                ID2.append(id[w + c])
                                IDpx.append(pxeda)
                            else:
                                cv.circle(image2, centro1, int(met_raio*1.5), (255, 0, 0), -1)
                                font = cv.FONT_HERSHEY_SIMPLEX
                                texto = f'{id[w]}, {id[w+c]}'
                                cv.putText(image2, texto, (int(centro1[0]-20), int(centro1[1] - 15)), font, .5, (0, 0, 0), 2)
                                cv.putText(image2, f'{int(pxeda)}', (int(centro1[0] - 10), int(centro1[1] + 5)), cv.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                                light = 'white'
                                if pxeda_b > pxeda_g and pxeda_b > pxeda_r:
                                    light = 'blue'
                                if pxeda_g > pxeda_b and pxeda_g > pxeda_r:
                                    light = 'green'
                                if pxeda_r > pxeda_b and pxeda_r > pxeda_g:
                                    light = 'red'

                                cv.putText(image2, f'Abs: {light}', (int(centro1[0] - 27), int(centro1[1] + 20)), cv.FONT_HERSHEY_SIMPLEX, .4, (51, 255, 0), 1)

                                ID1.append(id[w])
                                ID2.append(id[w + c])
                                IDpx.append(pxeda)



    df['ID1'] = ID1
    df['ID2'] = ID2
    df['pixel'] = IDpx


    df2 = df2.append(df)


    cv.imshow('image2', image2)
    cv.waitKey(0)
    cv.destroyAllWindows()



    resp_2 = input(str('Save the processed image? [Y]/[N]: ')).upper()
    if resp_2 == 'Y':
        pasta_image = '/Micro/Placa_Imagem/'
        cv.imwrite(raiz_path + pasta_image + title, image2)

    resp = input(str('Save the data? [Y]/[N]: ')).upper()
    if resp == 'Y':
        df2.to_csv('pxeda.csv', index=False)
        data = pd.read_csv('combinado.csv')
        df3 = pd.DataFrame(data)

        micro_zero = pd.DataFrame(columns=['ID1', 'ID2', 'pixel'])


        for c in range(0, len(df)):
            if df.iloc[c, 2] == 0:
                micro_zero = micro_zero.append(df.iloc[c, :], ignore_index=True)


        for c in range(0, len(df3)):
            for w in range(0, len(micro_zero)):
                if df3.iloc[c, 0] == micro_zero.iloc[w, 0] and df3.iloc[c, 3] == micro_zero.iloc[w, 1]:
                    df3.iloc[c, 6] = 0  # pixel
                    df3.iloc[c, 7] = 0  # resultado'''

        df3.to_csv('combinado.csv', index=False)