import torch
from torch.distributions import multinomial
from matplotlib import pyplot as plt

fair_probs = torch.ones([6]) / 6

print("1\n",multinomial.Multinomial(1, fair_probs).sample())
print("2\n",multinomial.Multinomial(10, fair_probs).sample())
counts = multinomial.Multinomial(1000, fair_probs).sample()
print("3\n",counts / 1000)


counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

# 设置图像大小
plt.figure(figsize=(6,4.5))

# 显示中文标签（不然会报错）
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

for i in range(6):
    plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
plt.axhline(y=0.167, color='black', linestyle='dashed')
plt.gca().set_xlabel('实验次数')
plt.gca().set_ylabel('估算概率')
plt.legend()
plt.show()


