import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Classification import Classification
from Micro_eng import Microplate
from reg_tlc import regist_tlc


dados = pd.read_csv('cadastro.csv')
df = pd.DataFrame(dados)

data = pd.read_csv('combinado_2.csv')
df4 = pd.DataFrame(data)

df2 = {}

while True:
    print('-'*30, 'MENU', '-'*30)
    print('(1) - New compound registration\n'
          '(2) - Show registered compounds\n'
          '(3) - Make the binary combination\n'
          '(4) - Show the binary combinations\n'
          '(5) - Microplate image registration\n'
          '(6) - TLC image registration\n'
          '(7) - Mixture classification\n'
          '(8) - Heatmap and statistics\n'
          '(9) - Exit')
    opc = int(input('Option: '))
    if opc == 1:
        id = len(df) + 1
        resp = 'M'
        while True:
            resp = str(input(f'Register the ID: \033[1;31m{id}\033[m? (Y) or (N): ')).upper()
            if resp == 'Y':
                df2['ID'] = len(df) + 1
                df2['Nome'] = str(input('Commercial name of the compound: '))
                df2['Smile'] = str(input('SMILE representation: '))
                df = df.append(df2, ignore_index=True)
                df.to_csv('cadastro.csv', index=False)
                break
            else:
                break

    elif opc == 2:
        print('Showing registered compounds...')
        print('')
        print(f'\033[1;033m{df.loc[:, "ID":"Nome"].to_string(index=False)}\033[m')
        print('')

    elif opc == 4:
        print('Showing registered combinations...')
        print('')
        print(df4.loc[:, ['ID1', 'Nome1', 'ID2', 'Nome2']].to_string(index=False))
        print('')
        print(f'Total: {len(df4)} combinations')

    elif opc == 3:
        if df4.loc[len(df4)-1, 'ID2'] < df.loc[len(df)-1, 'ID']:
            print('Making the binary combinations...')
            comp_1 = {}
            comp_2 = {}
            mixture = pd.DataFrame(columns=['ID1', 'Nome1', 'Smile1', 'ID2', 'Nome2', 'Smile2', 'pixel', 'resultado'])

            i = 0
            for c in range(df4.loc[len(df4) - 1, 'ID2'], df.loc[len(df) - 1, 'ID']):
                n = df.loc[len(df) - 1, 'ID'] - df4.loc[len(df4) - 1, 'ID2']
                for h in range(0, len(df)-n+i):
                    comp_1 = df.iloc[h, :]
                    comp_2 = df.iloc[c, :]
                    comp_1.rename({'ID': 'ID1', 'Nome': 'Nome1', 'Smile': 'Smile1'}, inplace=True)
                    comp_2.rename({'ID': 'ID2', 'Nome': 'Nome2', 'Smile': 'Smile2'}, inplace=True)
                    mixture = mixture.append({**comp_1, **comp_2}, ignore_index=True)
                i += 1
            df4 = df4.append(mixture)
            df4 = df4.sort_values(['ID1', 'ID2'])
            resp_6 = str(input('Save? [Y]/[N]: ')).upper()
            if resp_6 == 'Y':
                df4.to_csv('combinado_2.csv', index=False)

        else:
            print("There aren't new compounds to be added!")

    elif opc == 7:
        resp_4 = int(input('           MIXTURE CLASSIFICATION\n'
                           '(1) Before irradiation\n'
                           '(2) After irradiation\n'
                           'Enter the option: '))
        print('-' * 55)

        resp_5 = str(input('Automatic classification? [Y]/[N]: ')).upper()

        Classification(resp_4, resp_5)


    elif opc == 5:
        am = int(input('How many samples are there?: '))
        title = str(input('Enter the name of the photo (with extension: jpg, png, etc): '))

        Microplate(am, title)

    elif opc == 6:
        num_spot = int(input('How many spot are?: '))
        regist_tlc(num_spot)

    elif opc == 8:
        cont_uncolor = 0
        cont_colorimetric_reaction = 0
        cont_EDA_reaction = 0
        cont_hit = 0

        for c in range(0, len(df4)):
            if df4.iloc[c, 6] > 20:
                if df4.iloc[c, 7] == 1:
                    df4.iloc[c, 7] = -1
                    cont_colorimetric_reaction += 1
                if df4.iloc[c, 7] == 2:
                    df4.iloc[c, 7] = 1
                    cont_hit += 1
                if df4.iloc[c, 7] == 3:
                    df4.iloc[c, 7] = 2
                    cont_EDA_reaction += 1
                    cont_hit += 1
            else:
                df4.iloc[c, 7] = 0
                cont_uncolor += 1

        print(
            f'\nTotal: {len(df4)} mixtures.\n'
            f'Colorless: {cont_uncolor}({(cont_uncolor * 100 / len(df4)):.1f}%) mixtures.')
        print(f'Colorimetric reaction: {cont_colorimetric_reaction}({(cont_colorimetric_reaction * 100 / len(df4)):.1f}%) mixtures.')
        print(
            f'hit-EDA: {cont_hit}({(cont_hit * 100 / len(df4)):.1f}%) mixtures, of which {cont_EDA_reaction}({(cont_EDA_reaction * 100 / len(df4)):.1f}%) are reactive!\n')

        df3 = df4.pivot('ID2', 'ID1', 'resultado')
        # df2 = df.pivot('ID2', 'ID1', 'pixel')

        cmap = sns.diverging_palette(240, 0, s=80, l=45, as_cmap=False, center='light', n=5)

        ax = sns.heatmap(df3, cmap=cmap, center=0, cbar_kws={"shrink": .4})
        plt.tight_layout()
        ax.set_xlabel('ID1', fontdict={'fontsize': 10, 'fontweight': 'bold'})
        ax.set_ylabel('ID2', fontdict={'fontsize': 10, 'fontweight': 'bold'})
        colorbar = ax.collections[0].colorbar
        colorbar.set_ticks([-0.7, 0, 0.8, 1.6])
        colorbar.set_ticklabels(['Colorimetric Reaction', 'Colorless Mixture', 'Hit-EDA', 'EDA-Reaction'])

        plt.show()

    elif opc == 9:
        print('\033[1;033mBye bye!!! :)\033[m')
        break
