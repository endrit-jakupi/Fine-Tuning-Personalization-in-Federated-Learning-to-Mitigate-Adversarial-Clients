import subprocess



f_values = [9,6,3,0]
m_values = [16,48, 128]
alpha_values = [3.0,10.0,100.0]

for m in m_values:
    for alpha in alpha_values:
        for f in f_values:
            try:
                command = f"python3 main.py --n 20 --m {m} --test_m 512 --T 500 --f {f} --dataset phishing --model logistic_phishing --lr 0.1 --heterogeneity dirichlet_phishing --batch_size {m} --alpha {alpha} --nb_run 5 --nb_main_client 1 --attack SF --eval_every 20"
                subprocess.run(command, shell=True)
            except Exception as e:
                print("Error with values", f, alpha, e)

