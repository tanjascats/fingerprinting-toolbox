import json
import matplotlib.pyplot as plt
import seaborn


def main():
    with open('robustness/horizontal/breast_cancer_w/false_miss_l32_g1.0_x1.json', 'r') as infile:
        res = json.load(infile)
    with open('robustness/horizontal/breast_cancer_w/false_miss_l32_g1.11_x1.json', 'r') as infile:
        res2 = json.load(infile)
    with open('robustness/horizontal/breast_cancer_w/false_miss_l32_g2.5_x1.json', 'r') as infile:
        res3 = json.load(infile)
    print(res)
    plt.plot(res.keys(), res.values())
    plt.plot(res2.keys(), res2.values())
    plt.plot(res3.keys(), res3.values())
    plt.show()


if __name__ == '__main__':
    main()