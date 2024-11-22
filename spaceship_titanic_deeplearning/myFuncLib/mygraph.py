import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def classification_scatter(DataDF,x,y,hue,xlabel,ylabel,s=60,
                            size=None,sizes=None,palette='Spectral'):
    # 산점도 그래프 그리기
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(x=x, y=y, hue=hue, size=size, data=DataDF, 
                            sizes=sizes,s=s, palette=palette, alpha=0.6, edgecolor='w')

    # 그래프 꾸미기
    title = f'{hue} {x} & {y} Scatter'
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 범례 조정
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(handles, labels, title=hue, title_fontsize='13', fontsize='11', loc='upper right')

    plt.tight_layout()
    plt.show()
    
def draw_linear_scatter(SR1,SR2,attr_list=['title','x_label','y_label']):
    x_sr = np.array(SR1)
    y_sr = np.array(SR2)

    # 1차원 다항식 피팅 (즉, 직선)
    slope, intercept = np.polyfit(x_sr, y_sr, 1)

    # 트렌드 라인 y 값 계산
    trend_line = slope * x_sr + intercept

    # Scatter 그리기
    plt.figure(figsize=(12,8))
    plt.scatter(x_sr,y_sr,label='Perch',alpha=0.7, color='royalblue')
    plt.plot(x_sr, trend_line, color='coral', linestyle='--', label='Trend Line')
    plt.title(attr_list[0])
    plt.xlabel(attr_list[1])
    plt.ylabel(attr_list[2])
    plt.legend()
    plt.grid()
    plt.show()

def draw_plot_graph(data_dict_list,title,xlabel,ylabel,xline=None,yline=None):
    for data_dict in data_dict_list:
        x_list = list(data_dict.keys())
        y_list = list(data_dict.values())
        plt.plot(x_list,y_list,color='royalblue',marker='.')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if yline:
            for y in yline:
                plt.axvline(y, 0, 1, color='coral', linestyle='--')
        if xline:
            for x in xline:
                plt.axvline(x, 0, 1, color='coral', linestyle='--')

        plt.grid()
        plt.show()