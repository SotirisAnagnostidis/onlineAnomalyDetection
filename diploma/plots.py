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
    
    
def plot_results(em_algorithm, legend=True):
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
    if legend:
        plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2, sharex=ax)
    for i in range(len(em_algorithm.get_gammas())):
        plt.plot(x, em_algorithm.get_gammas()[i], color=colors[i % len(colors)], label='gamme ' + str(i))
    plt.ylabel('estimated weight')
    if legend:
        plt.legend()
    plt.grid()
    
    plt.subplot(3, 1, 3, sharex=ax)
    plt.plot(x, em_algorithm.get_likelihood())
    plt.ylabel('likelihood')
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    
    rcParams['figure.figsize'] = 16, 9

def _plot_category(xx, yy, Z, category, em):
    for i, point in enumerate(np.c_[xx.ravel(), yy.ravel()]):
        Z[i] = em.score_anomaly_for_category(point, category)

    Z = Z / np.max(Z)

    Z = Z.reshape(xx.shape)


    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    for i, lambda_i in enumerate(em.lambdas):
            plt.scatter(lambda_i[0], lambda_i[1], s=em.gammas[i]*1000, linewidth=4, color='red', marker='x')

    
def plot_category(category, em, limits_x=[-5,105], limits_y=[-5,105], number_of_points=50):
    # plot the level sets of the decision function
    xx, yy = np.meshgrid(np.linspace(limits_x[0], limits_x[1], number_of_points, dtype=np.int64),
                         np.linspace(limits_y[0], limits_y[1], number_of_points, dtype=np.int64))
    Z = np.zeros(len(xx)*len(xx))
    
    _plot_category(xx, yy, Z, category, em)

    plt.show()
    
def plot_all_categories(em, limits_x=[-5,105], limits_y=[-5,105], number_of_points=50):
    xx, yy = np.meshgrid(np.linspace(limits_x[0], limits_x[1], number_of_points, dtype=np.int64),
                         np.linspace(limits_y[0], limits_y[1], number_of_points, dtype=np.int64))
    
    rcParams['figure.figsize'] = 16, 4*(int((em.n_clusters -1)/2) + 1)
    for i in range(em.n_clusters):
        Z = np.zeros(len(xx)*len(xx))
        plt.subplot(int((em.n_clusters -1)/2) + 1, 2, i + 1)
        plt.title('Cluster' + str(i))
        _plot_category(xx, yy, Z, i, em)

    plt.tight_layout()
    plt.show()
    rcParams['figure.figsize'] = 16, 9
        