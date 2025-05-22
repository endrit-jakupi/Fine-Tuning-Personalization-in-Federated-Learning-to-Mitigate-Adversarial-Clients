import subprocess
import time


f_values = [0, 3, 6, 9]
m_values = [8, 16]
alpha_values = [1.0, 3.0, 100]


for m in m_values:
    for alpha in alpha_values:
        for f in f_values:
            try:
                command = f"python3 main.py --dataset mnist --heterogeneity dirichlet_mnist --n 20 --m {m} --f {f} --T 100 --model cnn --lr 0.05 --attack NS --batch_size 64 --nb_main_client 2 --nb_run 5 --alpha {alpha} --nb_classes 10"
                subprocess.run(command, shell=True)
            except Exception as e:
                print("Error with values", f,alpha, e)
            
