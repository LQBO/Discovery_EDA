import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from BaselineRemoval import BaselineRemoval
from scipy.signal import find_peaks
import os


def regist_tlc(num_spot):

    def Baseline_correction(y):
        out_images_1 = np.array(y)
        polynomial_degree = 2
        input_array_1 = out_images_1
        baseObj_1 = BaselineRemoval(input_array_1)
        Modpoly_output_1 = baseObj_1.ModPoly(polynomial_degree)
        return Modpoly_output_1.tolist()


    def mulipleReplace(text):
        for char in " []":
            text = text.replace(char, "")
        return text


    ID_spot_df = pd.DataFrame(columns=['ID1', 'ID2', 'foto', 'spot', 'x', 'y'])
    ID1 = []
    ID2 = []
    spot = []

    for c in range(0, num_spot):
        print('-' * 20, f'SPOT-{c + 1}', '-' * 20)
        spot.append(c + 1)
        ID1.append(int(input(f'Enter the first ID sample for SPOT-{c + 1}: ')))
        ID2.append(int(input(f'Enter the second ID sample for SPOT-{c + 1}: ')))
    ID_spot_df['spot'] = spot
    ID_spot_df['ID1'] = ID1
    ID_spot_df['ID2'] = ID2
    pasta_raiz = os.path.dirname(os.path.abspath(__file__))

    resp_1 = int(input('(1) TLC - Before irradiation\n(2) TLC - After irradiation\nEnter the option: '))
    print('-' * 55)

    if resp_1 == 1:
        print('TLC - EDA hit')
        print('-' * 55)
        #pasta = 'C:/Users/Rodrigo/PycharmProjects/EDA/TLC/'
        pasta = '/TLC/'

    if resp_1 == 2:
        print('TLC - EDA reaction')
        print('-' * 55)
        #pasta = str('C:/Users/Rodrigo/PycharmProjects/EDA/TLC_rx/')
        pasta = '/TLC_rx/'

    title = str(input('Enter the TLC photo name (with extension: jpg, png, etc): '))

    ID_spot_df['foto'] = title

    # importando imagem
    image = cv.imread(pasta_raiz + pasta + title)
    scale_percent = 70  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

    print('Resized Dimensions : ', image.shape)
    cv.imshow("Resized image", image)
    cv.waitKey(0)

    # gray scale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_inv = cv.bitwise_not(gray)

    # converter para hsv
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    # limiarizando na saturação (s)
    th, limiar = cv.threshold(s, 175, 255, cv.THRESH_BINARY_INV)

    kernal = np.ones((3, 3), np.uint8)

    opening = cv.morphologyEx(limiar, cv.MORPH_OPEN, kernal)

    # encontrando os contornos
    cnts, _ = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # print(f'Número de contornos: {str(len(cnts))}')

    # ordena e pega o maior contorno:
    cnts = sorted(cnts, key=cv.contourArea)
    cnt = cnts[-2]
    image2 = image.copy()

    # pegando os pontos dos cantos:
    arclen = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.05 * arclen, True)
    # cv.drawContours(image2, [cnt], -1, (255, 0, 0), 1, cv.LINE_AA)
    cv.drawContours(image2, [approx], -1, (0, 0, 255), 1, cv.LINE_AA)

    for c in range(0, 4):
        cv.putText(image2, f'P{c}', (approx[c][0][0], approx[c][0][1]), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)

    cv.imshow('detected', image2)
    cv.waitKey(0)

    x1 = []
    y1 = []
    x2 = []
    y2 = []
    xl1 = []
    xl2 = []
    yl1 = []
    yl2 = []
    pltx1 = []
    pltx2 = []
    pltx3 = []
    plty1 = []
    plty2 = []
    plty3 = []
    b = []
    a = []
    px = {}
    centros_x = {}
    centros_y = {}
    n = 300

    #print(approx)

    tam_0a1 = ((approx[0][0][0] - approx[1][0][0]) ** 2 + (approx[0][0][1] - approx[1][0][1]) ** 2) ** 0.5
    tam_1a2 = ((approx[1][0][0] - approx[2][0][0]) ** 2 + (approx[1][0][1] - approx[2][0][1]) ** 2) ** 0.5

    if tam_0a1 > tam_1a2:
        p0 = 0
        p1 = 1
        p2 = 2
        p3 = 3
    else:
        p0 = 3
        p1 = 0
        p2 = 1
        p3 = 2

    for c in range(0, num_spot):
        x1.append(approx[p0][0][0] + abs(approx[p0][0][0] - (approx[p3][0][0])) / (2 * num_spot) + c * abs(
            approx[p0][0][0] - (approx[p3][0][0])) / num_spot)
        y1.append(approx[p0][0][1] + abs(approx[p0][0][1] - (approx[p3][0][1])) / (2 * num_spot) + c * abs(
            approx[p0][0][1] - (approx[p3][0][1])) / num_spot)
        x2.append(approx[p1][0][0] + abs(approx[p1][0][0] - (approx[p2][0][0])) / (2 * num_spot) + c * abs(
            approx[p1][0][0] - (approx[p2][0][0])) / num_spot)
        y2.append(approx[p1][0][1] + abs(approx[p1][0][1] - (approx[p2][0][1])) / (2 * num_spot) + c * abs(
            approx[p1][0][1] - (approx[p2][0][1])) / num_spot)
        xl1.append(approx[p0][0][0] + abs(approx[p0][0][0] - (approx[p3][0][0])) / num_spot + c * abs(
            approx[p0][0][0] - (approx[p3][0][0])) / num_spot)
        yl1.append(approx[p0][0][1] + abs(approx[p0][0][1] - (approx[p3][0][1])) / num_spot + c * abs(
            approx[p0][0][1] - (approx[p3][0][1])) / num_spot)
        xl2.append(approx[p1][0][0] + abs(approx[p1][0][0] - (approx[p2][0][0])) / num_spot + c * abs(
            approx[p1][0][0] - (approx[p2][0][0])) / num_spot)
        yl2.append(approx[p1][0][1] + abs(approx[p1][0][1] - (approx[p2][0][1])) / num_spot + c * abs(
            approx[p1][0][1] - (approx[p2][0][1])) / num_spot)
        coef_ang = (y2[c] - y1[c]) / (x2[c] - x1[c])
        coef_lin = y1[c] - (coef_ang * x1[c])
        b.append(coef_ang)
        a.append(coef_lin)
        tam_c = ((x2[c] - x1[c]) ** 2 + (y2[c] - y1[c]) ** 2) ** (1 / 2)
        raio = tam_c / (2 * n)
        raio_y = abs(approx[p0][0][1] - (approx[p3][0][1])) / (2 * num_spot)
        jan_y = int(raio_y * 0.75)
        cv.line(image2, (np.float16(xl1[c]), np.float16(yl1[c])), (np.float16(xl2[c]), np.float16(yl2[c])), (0, 100, 255))

        for h in range(0, n):
            centros_x[c, h] = (x1[0] + tam_c / (2 * n) + h * (tam_c / (n)))
            centros_y[c, h] = a[c] + b[c] * centros_x[c, h]
            cv.circle(image2, (np.uint16(centros_x[c, h]), np.uint16(centros_y[c, h])), int(raio * 0.9), (255, 0, 0), -1)
            janela = gray_inv[
                     int(np.uint16(centros_y[c, h]) - np.uint16(jan_y)):int(np.uint(centros_y[c, h]) + np.uint16(jan_y)),
                     int(np.uint16(centros_x[c, h]) - np.uint16(raio)):int(np.uint16(centros_x[c, h]) + np.uint16(raio))]
            px[c, h] = int(np.mean(janela))


    for c in range(0, num_spot):
        for h in range(0, n):
            pltx1.append(h)
            plty1.append(px[c, h])
        y_base = Baseline_correction(plty1)
        ID_spot_df.iloc[c, 4] = str(pltx1)
        ID_spot_df.iloc[c, 5] = str(y_base)
        pltx1.clear()
        plty1.clear()



    cv.imshow('detected', image2)
    cv.waitKey(0)


    d = num_spot // 3
    r = num_spot % 3
    ax = {}

    if d == 0:
        for i in range(0, r):
            if r == 2:
                j = i + 211
            elif r == 1:
                j = i + 111
            ax[i] = plt.subplot(j)
            tex_x = str(ID_spot_df.iloc[i, 4])
            tex_x = mulipleReplace(tex_x).split(sep=',')
            x = [int(val) for val in tex_x]
            tex_y = str(ID_spot_df.iloc[i, 5])
            tex_y = mulipleReplace(tex_y).split(sep=',')
            y = [float(val) for val in tex_y]
            out_images_1 = np.array(y)
            ax[i].plot(x, out_images_1, label=f'ID: {ID1[i]}-{ID2[i]}')
            ax[i].set_xlabel(f'spot {1 + i}')
            ax[i].set_ylabel('count')
            ax[i].set_xlim([30, 290])
            ax[i].legend()
            peaks_1, _ = find_peaks(out_images_1, height=15, distance=5, width=5, prominence=2)
            ax[i].plot(peaks_1, out_images_1[peaks_1], 'kx')
            for p in range(0, len(peaks_1)):
                ax[i].annotate(f'{peaks_1[p]}', xy=(peaks_1[p], out_images_1[peaks_1[p]] + 5))
        plt.show()

    for c in range(0, d):
        for h in range(0, 3):
            j = h+311
            ax[h] = plt.subplot(j)
            tex_x = str(ID_spot_df.iloc[3*c + h, 4])
            tex_x = mulipleReplace(tex_x).split(sep=',')
            x = [int(val) for val in tex_x]
            tex_y = str(ID_spot_df.iloc[3*c + h, 5])
            tex_y = mulipleReplace(tex_y).split(sep=',')
            y = [float(val) for val in tex_y]
            out_images_1 = np.array(y)
            ax[h].plot(x, out_images_1, label=f'ID: {ID1[3*c + h]}-{ID2[3*c + h]}')
            ax[h].set_xlabel(f'spot {3*c+h+1}')
            ax[h].set_ylabel('count')
            ax[h].set_ylim([-5, 100])
            ax[h].set_xlim([30, 290])
            ax[h].legend()
            peaks_1, _ = find_peaks(out_images_1, height=15, distance=5, width=5, prominence=2)
            ax[h].plot(peaks_1, out_images_1[peaks_1], 'kx')
            for p in range(0, len(peaks_1)):
                ax[h].annotate(f'{peaks_1[p]}', xy=(peaks_1[p], out_images_1[peaks_1[p]] + 5))
        plt.show()

        if c == d-1 and r != 0:
            for i in range(0, r):
                if r == 2:
                    j = i + 211
                elif r == 1:
                    j = i + 111
                ax[i] = plt.subplot(j)
                tex_x = str(ID_spot_df.iloc[3*d+i, 4])
                tex_x = mulipleReplace(tex_x).split(sep=',')
                x = [int(val) for val in tex_x]
                tex_y = str(ID_spot_df.iloc[3*d+i, 5])
                tex_y = mulipleReplace(tex_y).split(sep=',')
                y = [float(val) for val in tex_y]
                out_images_1 = np.array(y)
                ax[i].plot(x, out_images_1, label=f'ID: {ID1[3*c + h]}-{ID2[3*c + h]}')
                ax[i].set_xlabel(f'spot {3 * d + 1 + i}')
                ax[i].set_ylabel('count')
                ax[i].set_xlim([30, 290])
                ax[i].legend()
                peaks_1, _ = find_peaks(out_images_1, height=15, distance=5, width=5, prominence=2)
                ax[i].plot(peaks_1, out_images_1[peaks_1], 'kx')
                for p in range(0, len(peaks_1)):
                    ax[h].annotate(f'{peaks_1[p]}', xy=(peaks_1[p], out_images_1[peaks_1[p]] + 5))

        plt.show()
    cv.destroyAllWindows()

    resp_2 = input(str('Save? [Y]/[N]: ')).upper()

    if resp_2 == 'Y':
        if resp_1 == 1:
            dados = pd.read_csv('tlcgram.csv')
            df = pd.DataFrame(dados)
            df = df.append(ID_spot_df, ignore_index=True)
            pasta_image = '/TLC/TLC_Imagem/'
            cv.imwrite(pasta_raiz + pasta_image + title, image2)
            df.to_csv('tlcgram.csv', index=False)
            print('\033[1;036mSAVED!!!\033[m')

        if resp_1 == 2:
            dados = pd.read_csv('tlc_rx.csv')
            df = pd.DataFrame(dados)
            df = df.append(ID_spot_df, ignore_index=True)
            pasta_image = '/TLC_rx/TLC_Imagem/'
            cv.imwrite(pasta_raiz + pasta_image + title, image2)
            df.to_csv('tlc_rx.csv', index=False)
            print('\033[1;015mSuccessfully saved!!!\033[m')

