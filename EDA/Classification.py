import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from scipy.signal import find_peaks
import os


def Classification(resp_4, resp_5):


    def mulipleReplace(text):
        for char in " []":
            text = text.replace(char, "")
        return text


    comb = pd.read_csv('combinado_3.csv')
    #comb = pd.read_csv('combinado.csv')
    comb_df = pd.DataFrame(comb)
    null_comb = pd.DataFrame(columns=['ID1', 'ID2', 'resultado'])
    pasta_raiz = os.path.dirname(os.path.abspath(__file__))

    for c in range(0, len(comb_df)):
        if pd.isnull(comb_df.iloc[c, 7]):
            null_comb = null_comb.append(comb_df.iloc[c, [0, 3, 7]], ignore_index=True)
            classif = 'Empty'

    if resp_4 == 1:
        print('BEFORE IRRADIATION')
        print('-' * 55)
        micro = pd.read_csv('pxeda.csv')
        micro_df = pd.DataFrame(micro)
        micro_df2 = pd.DataFrame(columns=['ID1', 'ID2', 'pixel'])
        for c in range(0, len(micro_df)):
            if micro_df.iloc[c, 2] > 20:
                micro_df2 = micro_df2.append(micro_df.iloc[c, :], ignore_index=True)

    if resp_4 == 2:
        print('AFTER IRRADIATION')
        print('-' * 55)
        pasta_2 = '/TLC_rx/'
        tlc_rx = pd.read_csv('tlc_rx.csv')
        tlc_rx_df = pd.DataFrame(tlc_rx)
        micro_df2 = pd.DataFrame(columns=['ID1', 'ID2', 'resultado'])
        for c in range(0, len(comb_df)):
            if comb_df.iloc[c, 6] > 20:
                if comb_df.iloc[c, 7] != 0:
                    micro_df2 = micro_df2.append(comb_df.iloc[c, [0, 3, 7]], ignore_index=True)


    pasta_1 = '/TLC/'
    tlc = pd.read_csv('tlcgram.csv')
    tlc_df = pd.DataFrame(tlc)

    if resp_5 == 'Y':
        if len(null_comb) == 0:
            print("There aren't mixtures to be classified!")
        else:
            for esc in range(0, len(null_comb)):
                ID1_escolha = null_comb.iloc[esc, 0]
                ID2_escolha = null_comb.iloc[esc, 1]
                for c in range(0, len(tlc_df)):
                    if ID1_escolha == tlc_df.iloc[c, 0] and ID1_escolha == tlc_df.iloc[c, 1]:
                        tex_y = str(tlc_df.iloc[c, 5])
                        tex_y = mulipleReplace(tex_y).split(sep=',')
                        y_1 = [float(val) for val in tex_y]
                        out_images_1 = np.array(y_1)

                    if ID2_escolha == tlc_df.iloc[c, 0] and ID2_escolha == tlc_df.iloc[c, 1]:
                        tex_y = str(tlc_df.iloc[c, 5])
                        tex_y = mulipleReplace(tex_y).split(sep=',')
                        y_2 = [float(val) for val in tex_y]
                        out_images_2 = np.array(y_2)

                    if resp_4 == 1:
                        if ID1_escolha == tlc_df.iloc[c, 0] and ID2_escolha == tlc_df.iloc[c, 1]:
                            tex_y = str(tlc_df.iloc[c, 5])
                            tex_y = mulipleReplace(tex_y).split(sep=',')
                            y_3 = [float(val) for val in tex_y]
                            out_images_3 = np.array(y_3)
                if resp_4 == 2:
                    for c in range(0, len(tlc_rx_df)):
                        if ID1_escolha == tlc_rx_df.iloc[c, 0] and ID2_escolha == tlc_rx_df.iloc[c, 1]:
                            tex_y = str(tlc_rx_df.iloc[c, 5])
                            tex_y = mulipleReplace(tex_y).split(sep=',')
                            y_3 = [float(val) for val in tex_y]
                            out_images_3 = np.array(y_3)



                peaks_1, _ = find_peaks(y_1, height=15, distance=5, width=5, prominence=2)
                peaks_2, _ = find_peaks(y_2, height=15, distance=5, width=5, prominence=2)
                peaks_3, _ = find_peaks(y_3, height=15, distance=5, width=5, prominence=2)

                # peaks compare:
                n_1 = len(peaks_1)
                n_2 = len(peaks_2)
                n_3 = len(peaks_3)

                new_peaks = {}
                remove_peaks = {}
                for j in range(0, n_3):
                    n = []
                    h = []
                    if n_1 == 0 and n_2 != 0:
                        for m in range(0, n_2):
                            if peaks_3[j] > 50:
                                if peaks_3[j] > peaks_2[m]:
                                    if peaks_3[j] - 14 > peaks_2[m]:
                                        n = peaks_3[j]
                                        new_peaks[n] = 'new'
                                    else:
                                        h = peaks_3[j]
                                        remove_peaks[h] = 'remove'
                                else:
                                    if peaks_3[j] + 14 < peaks_2[m]:
                                        n = peaks_3[j]
                                        new_peaks[n] = 'new'
                                    else:
                                        h = peaks_3[j]
                                        remove_peaks[h] = 'remove'
                    if n_1 != 0 and n_2 == 0:
                        for k in range(0, n_1):
                            if peaks_3[j] > 30:
                                if peaks_3[j] > peaks_1[k]:
                                    if peaks_3[j] - 14 > peaks_1[k]:
                                        n = peaks_3[j]
                                        new_peaks[n] = 'new'
                                    else:
                                        h = peaks_3[j]
                                        remove_peaks[h] = 'remove'
                                else:
                                    if peaks_3[j] + 14 < peaks_1[k]:
                                        n = peaks_3[j]
                                        new_peaks[n] = 'new'
                                    else:
                                        h = peaks_3[j]
                                        remove_peaks[h] = 'remove'

                    if n_1 == 0 and n_2 == 0:
                        if peaks_3[j] > 50:
                            n = peaks_3[j]
                            new_peaks[n] = 'new'
                    if n_1 != 0 and n_2 != 0:
                        for k in range(0, n_1):
                            for m in range(0, n_2):
                                if peaks_3[j] > 50:
                                    if peaks_3[j] > peaks_1[k]:
                                        if peaks_3[j] - 14 > peaks_1[k]:
                                            n = peaks_3[j]
                                            new_peaks[n] = 'new'
                                        else:
                                            h = peaks_3[j]
                                            remove_peaks[h] = 'remove'
                                    else:
                                        if peaks_3[j] + 14 < peaks_1[k]:
                                            n = peaks_3[j]
                                            new_peaks[n] = 'new'
                                        else:
                                            h = peaks_3[j]
                                            remove_peaks[h] = 'remove'
                                    if peaks_3[j] > peaks_2[m]:
                                        if peaks_3[j] - 14 > peaks_2[m]:
                                            n = peaks_3[j]
                                            new_peaks[n] = 'new'
                                        else:
                                            h = peaks_3[j]
                                            remove_peaks[h] = 'remove'
                                    else:
                                        if peaks_3[j] + 14 < peaks_2[m]:
                                            n = peaks_3[j]
                                            new_peaks[n] = 'new'
                                        else:
                                            h = peaks_3[j]
                                            remove_peaks[h] = 'remove'

                keys_n = list(new_peaks.keys())
                keys_r = list(remove_peaks.keys())
                picos = keys_n.copy()
                for c in range(0, len(keys_n)):
                    for d in range(0, len(keys_r)):
                        if keys_n[c] == keys_r[d]:
                            delete = keys_r[d]
                            picos.remove(delete)
                if resp_4 == 1:
                    if len(picos) == 0:
                        null_comb.iloc[esc, 2] = 2  # 2 - Hit-EDA
                        cont = 'HIT-EDA'
                    else:
                        null_comb.iloc[esc, 2] = 1  # 1 - Colorimetric Reaction
                        cont = 'COLORIMETRIC REACTION'

                if resp_4 == 2:
                    if len(picos) == 0:
                        null_comb.iloc[esc, 2] = 2  # 2 - Hit-EDA
                        cont = 'HIT-EDA'
                    else:
                        null_comb.iloc[esc, 2] = 3  # 3 - EDA Reaction
                        cont = 'EDA REACTION'

                print(f'ID: {null_comb.iloc[esc, 0]}-{null_comb.iloc[esc, 1]}, Peaks: {picos}, Classificated as {cont}')

            resp2 = str(input('Save? [Y]/[N]: ')).upper()
            if resp2 == 'Y':
                for c in range(0, len(comb_df)):
                    for d in range(0, len(null_comb)):
                        if comb_df.iloc[c, 0] == null_comb.iloc[d, 0] and comb_df.iloc[c, 3] == null_comb.iloc[d, 1]:
                            comb_df.iloc[c, 7] = null_comb.iloc[d, 2]

                comb_df.to_csv('combinado_3.csv', index=False)
                print('\033[1;31mSaved successfully\033[m')

    if resp_5 == 'N':
        print(micro_df2.to_string(index=False))
        ID1_escolha = int(input('Enter ID1 of the mixture: '))
        ID2_escolha = int(input('Enter ID2 of the mixture: '))
        for c in range(0, len(tlc_df)):
            if ID1_escolha == tlc_df.iloc[c, 0] and ID1_escolha == tlc_df.iloc[c, 1]:
                tex_x = str(tlc_df.iloc[c, 4])
                tex_y = str(tlc_df.iloc[c, 5])
                foto1 = tlc_df.iloc[c, 2]
                spot1 = tlc_df.iloc[c, 3]
                tex_x = mulipleReplace(tex_x).split(sep=',')
                tex_y = mulipleReplace(tex_y).split(sep=',')
                x_1 = [int(val) for val in tex_x]
                y_1 = [float(val) for val in tex_y]
                out_images_1 = np.array(y_1)

            if ID2_escolha == tlc_df.iloc[c, 0] and ID2_escolha == tlc_df.iloc[c, 1]:
                tex_x = str(tlc_df.iloc[c, 4])
                tex_y = str(tlc_df.iloc[c, 5])
                foto2 = tlc_df.iloc[c, 2]
                spot2 = tlc_df.iloc[c, 3]
                tex_x = mulipleReplace(tex_x).split(sep=',')
                tex_y = mulipleReplace(tex_y).split(sep=',')
                x_2 = [int(val) for val in tex_x]
                y_2 = [float(val) for val in tex_y]
                out_images_2 = np.array(y_2)

            if resp_4 == 1:
                if ID1_escolha == tlc_df.iloc[c, 0] and ID2_escolha == tlc_df.iloc[c, 1]:
                    tex_x = str(tlc_df.iloc[c, 4])
                    tex_y = str(tlc_df.iloc[c, 5])
                    foto3 = tlc_df.iloc[c, 2]
                    spot3 = tlc_df.iloc[c, 3]
                    tex_x = mulipleReplace(tex_x).split(sep=',')
                    tex_y = mulipleReplace(tex_y).split(sep=',')
                    x_3 = [int(val) for val in tex_x]
                    y_3 = [float(val) for val in tex_y]
                    out_images_3 = np.array(y_3)

        if resp_4 == 2:
            for c in range(0, len(tlc_rx_df)):
                if ID1_escolha == tlc_rx_df.iloc[c, 0] and ID2_escolha == tlc_rx_df.iloc[c, 1]:
                    tex_x = str(tlc_rx_df.iloc[c, 4])
                    tex_y = str(tlc_rx_df.iloc[c, 5])
                    foto3 = tlc_rx_df.iloc[c, 2]
                    spot3 = tlc_rx_df.iloc[c, 3]
                    tex_x = mulipleReplace(tex_x).split(sep=',')
                    tex_y = mulipleReplace(tex_y).split(sep=',')
                    x_3 = [int(val) for val in tex_x]
                    y_3 = [float(val) for val in tex_y]
                    out_images_3 = np.array(y_3)

        image1 = cv.imread(str(pasta_raiz + pasta_1 + foto1))
        image2 = cv.imread(str(pasta_raiz + pasta_1 + foto2))
        if resp_4 == 1:
            image3 = cv.imread(str(pasta_raiz + pasta_1 + foto3))
        if resp_4 == 2:
            image3 = cv.imread(str(pasta_raiz + pasta_2 + foto3))


        linhas1, colunas1, canais1 = image1.shape
        R1 = cv.getRotationMatrix2D((colunas1 / 2, linhas1 / 2), 90, 1)
        image1 = cv.warpAffine(image1, R1, (colunas1, linhas1))

        linhas2, colunas2, canais2 = image2.shape
        R2 = cv.getRotationMatrix2D((colunas2 / 2, linhas2 / 2), 90, 1)
        image2 = cv.warpAffine(image2, R2, (colunas2, linhas2))

        linhas3, colunas3, canais3 = image3.shape
        R3 = cv.getRotationMatrix2D((colunas3 / 2, linhas3 / 2), 90, 1)
        image3 = cv.warpAffine(image3, R3, (colunas3, linhas3))

        cv.putText(image1, f'Spot {spot1}', (500, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 7)
        cv.putText(image2, f'Spot {spot2}', (500, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 7)
        cv.putText(image3, f'Spot {spot3}', (500, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)

        peaks_1, _ = find_peaks(y_1, height=15, distance=5, width=5, prominence=2)
        peaks_2, _ = find_peaks(y_2, height=15, distance=5, width=5, prominence=2)
        peaks_3, _ = find_peaks(y_3, height=15, distance=5, width=5, prominence=2)

        ax1 = plt.subplot(231)
        ax2 = plt.subplot(232)
        ax3 = plt.subplot(233)
        ax4 = plt.subplot(212)

        ax1.imshow(image1)
        ax1.set_title(f'ID: {ID1_escolha}')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.imshow(image2)
        ax2.set_title(f'ID: {ID2_escolha}')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.imshow(image3)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title(f'ID: {ID1_escolha}-{ID2_escolha}')
        ax4.plot(x_1, y_1, color='red', label=f'ID1: {ID1_escolha}')
        ax4.set_xlabel('pixel')
        ax4.set_ylabel('count')
        ax4.set_xlim([30, 290])
        ax4.set_ylim([-5, 100])
        ax4.plot(x_2, y_2, color='blue', label=f'ID2: {ID2_escolha}')
        ax4.plot(x_3, y_3, color='black', label=f'{ID1_escolha}, {ID2_escolha}')
        ax4.legend()
        for c in range(0, len(comb_df)):
            if comb_df.iloc[c, 0] == ID1_escolha and comb_df.iloc[c, 3] == ID2_escolha:
                if comb_df.iloc[c, 7] == 1:
                    classif = 'Colorimetric Reaction'
                if comb_df.iloc[c, 7] == 2:
                    classif = 'Hit EDA'
                if comb_df.iloc[c, 7] == 3:
                    classif = 'EDA - Reaction'
                ax4.text(0.005, 0.9, f'Classification: {classif}', transform=ax4.transAxes, color='red', fontsize=11,
                         fontweight='bold')


        ax4.plot(peaks_1, out_images_1[peaks_1], 'rx')
        ax4.plot(peaks_2, out_images_2[peaks_2], 'bx')
        ax4.plot(peaks_3, out_images_3[peaks_3], 'kx')

        for p in range(0, len(peaks_1)):
            ax4.annotate(f'{peaks_1[p]}', xy=(peaks_1[p], out_images_1[peaks_1[p]] + 5), color='red')
        for p in range(0, len(peaks_2)):
            ax4.annotate(f'{peaks_2[p]}', xy=(peaks_2[p], out_images_2[peaks_2[p]] + 5), color='blue')
        for p in range(0, len(peaks_3)):
            ax4.annotate(f'{peaks_3[p]}', xy=(peaks_3[p], out_images_3[peaks_3[p]] + 5), color='black')


        plt.show()


        resp2 = str(input('Save? [Y]/[N]: ')).upper()
        if resp2 == 'Y':
            for c in range(0, len(comb_df)):
                if comb_df.iloc[c, 0] == ID1_escolha and comb_df.iloc[c, 3] == ID2_escolha:
                    comb_df.iloc[c, 7] = int(input(f'\n         CLASSIFICATION\n'
                                                   f'(1) - Colorimetric reaction\n'
                                                   f'(2) - Hit-EDA\n'
                                                   f'(3) - EDA Reaction\n'
                                                   f'\nID: {ID1_escolha}-{ID2_escolha}: '))
                    if comb_df.iloc[c, 7] == 1:
                        print(f'Save as COLORIMETRIC REACTION')
                    elif comb_df.iloc[c, 7] == 2:
                        print(f'Save as HIT-EDA')
                    elif comb_df.iloc[c, 7] == 3:
                        print(f'Save as EDA REACTION')

            #comb_df.to_csv('combinado.csv', index=False)
            comb_df.to_csv('combinado_3.csv', index=False)


