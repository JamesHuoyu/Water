import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "/home/debian/water/TIP4P/2005/2020/rst/tetrahedral_order_parameter_Q.csv"
data = pd.read_csv(file_path)
q_values = data["Q"].values
plt.figure(figsize=(8, 5))
plt.hist(q_values, bins=300, density=True, alpha=0.7, color="blue")

# ideal_x, ideal_y = np.loadtxt(
#     "/home/debian/water/TIP4P/2005/2020/rst/Default Dataset.csv", delimiter=",", unpack=True
# )
# plt.plot(ideal_x, ideal_y, color="red", lw=2, label="Ideal Distribution")
plt.legend()
plt.xlabel("Tetrahedral Order Parameter Q")
plt.ylabel("Probability Density")
plt.title("Distribution of Tetrahedral Order Parameter Q")
plt.savefig("/home/debian/water/TIP4P/2005/2020/rst/q_distribution.png", dpi=300)
# plt.show()
