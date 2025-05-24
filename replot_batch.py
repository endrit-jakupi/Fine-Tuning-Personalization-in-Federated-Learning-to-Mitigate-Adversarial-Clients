import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json
import matplotlib
import seaborn as sns
matplotlib.rcParams['pdf.fonttype'] = 42

matplotlib.rcParams['ps.fonttype'] = 42

#plt.rcParams.update({'font.size': 22})


sns.set_theme()
sns.set(font_scale=1.3)


def stack_plots_m(folders, n, f, alpha, data= "mnist_bin/dirichlet_mnist_bin"):
    output_file = f'experiments/{data}/m_plot_n_{n}_f_{f}_alpha_{alpha}.png'
    results_list = []
    config_list = []
    f1_scores_list = []
    for folder in folders:
        with open(os.path.join(folder, 'results.pickle'), 'rb') as file:
            results_list.append(pickle.load(file))

        with open(os.path.join(folder, 'config.json'), 'r') as file:
            config_list.append(json.load(file))

        with open(os.path.join(folder, 'f1_scores.pickle'), 'rb') as file:
            f1_scores_list.append(pickle.load(file))
    
    
    lam_bar_list =   np.array([0, 0.2, 0.4, 0.6, 0.8, 1 ]) #np.array(config_list[0]['lams'])#1- np.array(config_list[0]['lams']) # 1 - \lambda

    #plt.errorbar(lam_bar, np.mean(results, axis = 0), yerr = np.std(results, axis = 0), fmt='o', ecolor='orangered', capsize=3 )
    for i in range(len(folders)):
        #plt.plot(lam_bar_list, np.mean(results_list[i], axis = 0), 'o--', label = f"f = {config_list[i]['f']}")
        plt.errorbar(lam_bar_list, np.mean(results_list[i][:,:,0], axis = (0)), yerr = np.std(results_list[i][:,:,0], axis = (0)) ,fmt= 'o--', label = f"m = {config_list[i]['m']}")
    
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Test accuracy on local dataset")

    plt.ylim(0, 100)
    plt.legend()
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()
    output_file_f1_score = f'experiments/{data}/f1_score_m_plot_n_{n}_f_{f}_alpha_{alpha}.png'
    for i in range(len(folders)):
        #plt.plot(lam_bar_list, np.mean(results_list[i], axis = 0), 'o--', label = f"f = {config_list[i]['f']}")
        plt.errorbar(lam_bar_list, np.mean(f1_scores_list[i][:,:,0], axis = (0)), yerr = np.std(f1_scores_list[i][:,:,0], axis = (0)) ,fmt= 'o--', label = f"m = {config_list[i]['m']}")

    plt.xlabel(r"$\lambda$")
    plt.ylabel("F1 scores on local dataset")

    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(output_file_f1_score, bbox_inches='tight')
    plt.show()
    plt.close()

def stack_plots_alpha(folders,n,f,m, data= "mnist_bin/dirichlet_mnist_bin"):
    
    output_folder = f'experiments/{data}/alpha_plot_n_{n}_f_{f}_m_{m}.png'
    results_list = []
    config_list = []
    f1_scores_list = []
    for folder in folders:
        with open(os.path.join(folder, 'results.pickle'), 'rb') as file:
            results_list.append(pickle.load(file))

        with open(os.path.join(folder, 'config.json'), 'r') as file:
            config_list.append(json.load(file))

        with open(os.path.join(folder, 'f1_scores.pickle'), 'rb') as file:
            f1_scores_list.append(pickle.load(file))
    lam_bar_list = np.array([0, 0.2, 0.4, 0.6, 0.8, 1 ]) #np.array(config_list[0]['lams'])#1- np.array(config_list[0]['lams']) # 1 - \lambda

    #plt.errorbar(lam_bar, np.mean(results, axis = 0), yerr = np.std(results, axis = 0), fmt='o', ecolor='orangered', capsize=3 )
    for i in range(len(folders)):
        #plt.plot(lam_bar_list, np.mean(results_list[i], axis = 0), 'o--', label = f"f = {config_list[i]['f']}")
        plt.errorbar(lam_bar_list, np.mean(results_list[i][:,:,0], axis = (0)), yerr = np.std(results_list[i][:,:,0], axis = (0)) ,fmt= 'o--', label = rf"$\alpha = {config_list[i]['alpha']}$")
    plt.xlabel(r"$\lambda$")
    #plt.ylabel("Test accuracy on local dataset")

    plt.ylim(0, 100)
    plt.legend()
    
    plt.savefig(output_folder, bbox_inches='tight')
    plt.show()
    plt.close()
    
    
    output_file_f1_score = f'experiments/{data}/f1_score_m_plot_n_{n}_f_{f}_m_{m}.png'
    for i in range(len(folders)):
        #plt.plot(lam_bar_list, np.mean(results_list[i], axis = 0), 'o--', label = f"f = {config_list[i]['f']}")
        plt.errorbar(lam_bar_list, np.mean(f1_scores_list[i][:,:,0], axis = (0)), yerr = np.std(f1_scores_list[i][:,:,0], axis = (0)) ,fmt= 'o--', label = rf"$\alpha = {config_list[i]['alpha']}$")

    plt.xlabel(r"$\lambda$")
    plt.ylabel("F1 scores on local dataset")

    plt.ylim(0, 1)
    plt.legend()
    
    plt.savefig(output_file_f1_score, bbox_inches='tight')
    plt.show()
    plt.close()


def stack_plots_byz(folders,n,m, alpha, data= "mnist_bin/dirichlet_mnist_bin"):
    output_folder = f'experiments/{data}/byz_plot_n_{n}_m_{m}_alpha_{alpha}.png'
    results_list = []
    config_list = []
    f1_scores_list = []
    for folder in folders:
        with open(os.path.join(folder, 'results.pickle'), 'rb') as file:
            results_list.append(pickle.load(file))

        with open(os.path.join(folder, 'config.json'), 'r') as file:
            config_list.append(json.load(file))
        
        with open(os.path.join(folder, 'f1_scores.pickle'), 'rb') as file:
            f1_scores_list.append(pickle.load(file))
    
    lam_bar_list =   np.array([0, 0.2, 0.4, 0.6, 0.8, 1 ]) #np.array(config_list[0]['lams'])#1- np.array(config_list[0]['lams']) # 1 - \lambda

    #plt.errorbar(lam_bar, np.mean(results, axis = 0), yerr = np.std(results, axis = 0), fmt='o', ecolor='orangered', capsize=3 )
    for i in range(len(folders)):
        #plt.plot(lam_bar_list, np.mean(results_list[i], axis = 0), 'o--', label = f"f = {config_list[i]['f']}")
        #plt.plot(lam_bar_list, results_list[i][0,:,0], 'o--', label = rf"$f = {config_list[i]['f']}$")
        #errors = np.std(results_list[i][:,:,0], axis = (0))*1.96/np.sqrt(5)
        #plt.plot(lam_bar_list, np.mean(results_list[i][:,:,0], axis = (0)), 'o--', label = rf"$f = {config_list[i]['f']}$")
        #plt.fill_between(lam_bar_list, np.mean(results_list[i][:,:,0], axis = (0)) - errors, np.mean(results_list[i][:,:,0], axis = (0)) + errors, alpha=0.2)
        
        plt.errorbar(lam_bar_list, np.mean(results_list[i][:,:,0], axis = (0)), yerr = np.std(results_list[i][:,:,0], axis = (0)) ,fmt= 'o--', label = rf"$f = {config_list[i]['f']}$")
    plt.xlabel(r"$\lambda$")
    #plt.ylabel("Test accuracy on local dataset")

    plt.ylim(0, 100)
    plt.legend()
    plt.savefig(output_folder, bbox_inches='tight')

    plt.show()
    plt.close()
    output_file_f1_score = f'experiments/{data}/f1_score_m_plot_n_{n}_m_{m}_alpha_{alpha}.png'
    for i in range(len(folders)):
        #plt.plot(lam_bar_list, np.mean(results_list[i], axis = 0), 'o--', label = f"f = {config_list[i]['f']}")
        plt.errorbar(lam_bar_list, np.mean(f1_scores_list[i][:,:,0], axis = (0)), yerr = np.std(f1_scores_list[i][:,:,0], axis = (0)) ,fmt= 'o--', label = f"f = {config_list[i]['f']}")

    plt.xlabel(r"$\lambda$")
    plt.ylabel("F1 scores on local dataset")

    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(output_file_f1_score, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    f_values = [0, 3, 6, 9]
    m_values = [8, 16]
    alpha_values = [1.0, 3.0, 100.0]


    for f in f_values:
        for alpha in alpha_values:
            stack_plots_m([f"experiments/mnist/dirichlet_mnist/NS/n_20_m_{m}_f_{f}_T_100_runs_5_alpha_{alpha}" for m in m_values], 20, f, alpha, data= "mnist/dirichlet_mnist/NS")
        
    for f in f_values:
        for m in m_values:
            stack_plots_alpha([f"experiments/mnist/dirichlet_mnist/NS/n_20_m_{m}_f_{f}_T_100_runs_5_alpha_{alpha}" for alpha in alpha_values], 20, f, m, data= "mnist/dirichlet_mnist/NS")

    for m in m_values:
        for alpha in alpha_values:
            stack_plots_byz([f"experiments/mnist/dirichlet_mnist/NS/n_20_m_{m}_f_{f}_T_100_runs_5_alpha_{alpha}" for f in f_values], 20, m, alpha, data = "mnist/dirichlet_mnist/NS")