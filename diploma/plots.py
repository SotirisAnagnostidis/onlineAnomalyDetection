from pylab import rcParams
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
import matplotlib.patches as mpatches

colors = ['blue', 'red', 'green', 'yellow']
styles = ['-','--',':','-.']


def plot_points(data, em):
    rcParams['figure.figsize'] = 16, 9
    data_hashable = [tuple(x) for x in data]
    total_points = len(data_hashable)

    values = np.vstack([list(x) for x in list(Counter(data_hashable).keys())])
    counts = np.array(list(Counter(data_hashable).values()))

    for i in range(len(values)):
        plt.scatter(values[i][0], values[i][1], s=counts[i]*10000/total_points, color='blue')
        
    for i, lambda_i in enumerate(em.lambdas):
        plt.scatter(lambda_i[0], lambda_i[1], s=em.gammas[i]*1000, linewidth=4, color='red', marker='x')

    blue_patch = mpatches.Patch(color='blue', label='Data points')
    red_patch = mpatches.Patch(color='red', label='Centers of Poisson')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()
    
    
def plot_results(em_algorithm):
    import matplotlib
    
    rcParams['figure.figsize'] = 16, 12
    rcParams['legend.loc'] = 'best'

    matplotlib.rcParams.update({'font.size': 16})

    x = range(1, len(em_algorithm.get_gammas()[1]) + 1)

    plt.title('Online EM results')

    ax = plt.subplot(3, 1, 1)

    for i in range(len(em_algorithm.get_lambdas())):
        for j in range(em_algorithm.dim):
            if j == 0:
                a = plt.plot(x, np.array(em_algorithm.get_lambdas()[i])[:,j], color=colors[i % len(colors)], 
                             linestyle=styles[j % len(styles)], label='Distribution ' + str(i))
            else:
                a = plt.plot(x, np.array(em_algorithm.get_lambdas()[i])[:,j], color=colors[i % len(colors)], 
                             linestyle=styles[j % len(styles)])
    plt.ylabel('lambda')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2, sharex=ax)
    for i in range(len(em_algorithm.get_gammas())):
        plt.plot(x, em_algorithm.get_gammas()[i], color=colors[i % len(colors)], label='gamme ' + str(i))
    plt.ylabel('estimated weight')
    plt.legend()
    plt.grid()
    
    plt.subplot(3, 1, 3, sharex=ax)
    plt.plot(x, em_algorithm.get_likelihood())
    plt.ylabel('likelihood')
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    rcParams['figure.figsize'] = 16, 9