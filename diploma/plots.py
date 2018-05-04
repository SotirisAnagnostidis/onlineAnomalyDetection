from pylab import rcParams
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter
import matplotlib.patches as mpatches

colors = ['blue', 'red', 'green', 'yellow']
styles = ['-','--',':','-.']


def plot_points(data, em=None):
    rcParams['figure.figsize'] = 16, 9
    data_hashable = [tuple(x) for x in data]
    total_points = len(data_hashable)

    values = np.vstack([list(x) for x in list(Counter(data_hashable).keys())])
    counts = np.array(list(Counter(data_hashable).values()))

    for i in range(len(values)):
        plt.scatter(values[i][0], values[i][1], s=counts[i]*10000/total_points, color='blue')
        
    if em:
        for i, lambda_i in enumerate(em.lambdas):
            plt.scatter(lambda_i[0], lambda_i[1], s=em.gammas[i]*1000, linewidth=4, color='red', marker='x')

        blue_patch = mpatches.Patch(color='blue', label='Data points')
        red_patch = mpatches.Patch(color='red', label='Centers of Poisson')
        plt.legend(handles=[red_patch, blue_patch], fontsize=18)
    else:
        blue_patch = mpatches.Patch(color='blue', label='Data points')
        plt.legend(handles=[blue_patch], fontsize=18)
    plt.show()
    
    
def plot_results(em_algorithm, legend=True):
    import matplotlib
    
    rcParams['figure.figsize'] = 16, 12
    rcParams['legend.loc'] = 'best'

    matplotlib.rcParams.update({'font.size': 16})

    x = range(0, len(em_algorithm.get_gammas()[1]))

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
        
def plot_for_host(host, em, data, limits_x=[-5,105], limits_y=[-5,105], number_of_points=50):
    rcParams['figure.figsize'] = 16, 9
    category = em.hosts[host]['category']
    
    plt.subplot(1,2,1)
      # plot the level sets of the decision function
    xx, yy = np.meshgrid(np.linspace(limits_x[0], limits_x[1], number_of_points, dtype=np.int64),
                         np.linspace(limits_y[0], limits_y[1], number_of_points, dtype=np.int64))
    Z = np.zeros(len(xx)*len(xx))
    
    for i, point in enumerate(np.c_[xx.ravel(), yy.ravel()]):
        Z[i] = em.score_anomaly_for_category(point, category)
        
    temp = np.array(Z, copy=True)  
        
    Z = Z / np.max(Z)

    Z = Z.reshape(xx.shape)


    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    
    data_for_host = data[np.where(data[:,len(data[0]) - 1] == host)]
    
    plt.scatter(data_for_host[:,0], data_for_host[:,1], color='green', s=50)
    plt.title('Compared to its category/cluster past')
    
    plt.subplot(1,2,2)

    Z1 = np.zeros(len(xx)*len(xx))
    
    for i, point in enumerate(np.c_[xx.ravel(), yy.ravel()]):
        Z1[i] = em.score_anomaly_for_category(point, category, host)

    Z1 = Z1 - temp / 2
    
    Z1 = Z1 / np.max(Z1)
    

    Z1 = Z1.reshape(xx.shape)


    plt.contourf(xx, yy, Z1, cmap=plt.cm.Blues_r)

    
    plt.scatter(data_for_host[:,0], data_for_host[:,1], color='green', s=50)
    plt.title('Compared to its own past')
    
    plt.tight_layout()
    plt.show()
    
    
    return sorted([(point, em.score_anomaly_for_category(point, category, host)) for point in data_for_host], key=lambda tup: tup[1])

def plot_parameter_updates(data, em):
    rcParams['figure.figsize'] = 16, 9
    data_hashable = [tuple(x) for x in data]
    total_points = len(data_hashable)

    values = np.vstack([list(x) for x in list(Counter(data_hashable).keys())])
    counts = np.array(list(Counter(data_hashable).values()))

    for i in range(len(values)):
        poisson_center = np.argmax(em.calculate_participation([values[i]]))
        plt.scatter(values[i][0], values[i][1], s=counts[i]*10000/total_points, color=colors[poisson_center])
        
    for i, lambda_i_updates in enumerate(em.get_lambdas()):
        steps = len(lambda_i_updates)
        for j, lambda_i in enumerate(lambda_i_updates):
            plt.scatter(lambda_i[0], lambda_i[1], s=em.gammas[i]*200, linewidth=4, color=colors[i])
            if j < steps - 1:
                plt.arrow(lambda_i_updates[j][0], lambda_i_updates[j][1], lambda_i_updates[j + 1][0] - lambda_i_updates[j][0], 
                          lambda_i_updates[j + 1][1] - lambda_i_updates[j][1], 
                          length_includes_head=True, head_width=0.3, color='black', alpha=1.0/(steps-j))
    
    # HARD CODED
    for i, lambda_i_updates in enumerate(em.get_lambdas()):
        plt.axes([.65, .70 - i *0.20, .23, .15])
        for j, lambda_i in enumerate(lambda_i_updates):
            plt.scatter(lambda_i[0], lambda_i[1], s=em.gammas[i]*200, linewidth=4, color=colors[i])
            if j < steps - 1:
                plt.arrow(lambda_i_updates[j][0], lambda_i_updates[j][1], lambda_i_updates[j + 1][0] - lambda_i_updates[j][0], 
                          lambda_i_updates[j + 1][1] - lambda_i_updates[j][1], 
                          length_includes_head=True, color='black', alpha=1.0/(steps-j))
    

    plt.show()