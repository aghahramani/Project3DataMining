__author__ = 'alighahramani'
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
"""0
	0-4 : 2
	0-1 : 10
	0-2 : 115
	0-3 : 22
	total_count : 159
	wrong : 149
	correct : 10
	accuracy : 0.062893081761
1
	total_count : 829
	1-3 : 54
	1-2 : 696
	1-4 : 13
	wrong : 777
	1-0 : 14
	correct : 52
	accuracy : 0.0627261761158
2
	2-3 : 108
	2-0 : 15
	2-1 : 62
	2-4 : 10
	total_count : 2999
	wrong : 195
	correct : 2804
	accuracy : 0.934978326109
3
	total_count : 918
	3-4 : 19
	3-2 : 765
	3-1 : 33
	3-0 : 5
	wrong : 822
	correct : 96
	accuracy : 0.104575163399
4
	4-0 : 3
	4-1 : 21
	4-2 : 192
	4-3 : 36
	total_count : 258
	wrong : 252
	correct : 6
	accuracy : 0.0232558139535
total_accuracy : 0.574859577765"""

if __name__ == '__main__':
    N = 2
    matplotlib.rcParams.update({'font.size': 12})
    BN = (21.35,20.81,19.71,21.50) # Tree , Reg , ModTree, Mod Reg
    AN = (18.32,14.78,16.46,13.95)

    T = (21.35 , 18.32)
    R = (20.81 , 14.78)
    MT = (19.71 , 16.46)
    MR = (21.50 , 13.95)


    ind =  np.arange(N)

    width = .15

    fig, ax = plt.subplots()

    rects1 = ax.bar(ind, T, width, color='r')
    rects2 = ax.bar(ind+width+0.02, MT, width, color='b')
    rects3 = ax.bar(ind+2*width+0.04, R, width, color='y')
    rects4 = ax.bar(ind+3*width+0.06, MR, width, color='g')

    ax.set_ylabel('RMSE')
    ax.set_title('RMSE for each model')
    ax.set_xticks(ind+width+0.16)
    ax.set_xticklabels( ('Before Label Normalization', 'After Label Normalization'))
    ax.legend( (rects1[0], rects2[0],rects3[0] , rects4[0]), ('Base Line Decision Tree', 'Modified Decision Tree','Base Line Linear Regression' ,
    'Modified Linear Regression') )

    def autolabel(rects):
    # attach some text labels
        for rect in rects:
            height = rect.get_height()
            print height
            ax.text(rect.get_x()+rect.get_width()/2., height+0.15, '%.2f'%(height),
                    ha='center', va='bottom' , )
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    plt.show()
