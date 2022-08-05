# import this file to run the final experiments
# Tema selecto experiments
# PARAMETER MATRIX FROM EXCEL
dcs_list = [3, 3, 5, 10, 15]
num_commodities_list = [5, 5, 5, 5, 5]
customers_list = [5, 10, 15, 20, 25]
orders_per_day_list = [3, 5, 5, 10, 15]
dcs_per_customer_list = [2, 3, 5, 5, 5]

# OTHER PARAMS AND UTILS.
num_steps = 10
num_episodes = 100
num_experiments = len(dcs_list)
