import matplotlib.pyplot as plt

# Data
years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
population = [4496232, 4581642, 4706325, 4729185, 4833386, 5115316, 5277619, 5435273, 5584743, 5735796, 5882916,
              6024463, 6162433, 6296786]


def main():
    # Plotting
    plt.figure(figsize=(16, 8))
    plt.bar(years, population, color='skyblue')

    # Labeling
    plt.xlabel('Rok', fontsize=20)
    plt.ylabel('Počet osobních vozidel [v miliónech] ', fontsize=20)
    plt.title('Počet osobních vozidel v České Republice', fontsize=20)

    # Adjust ticks fontsize
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Annotating each bar with its value
    for i, txt in enumerate(population):
        plt.annotate(txt, (years[i], population[i]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=12)  # Adjusted fontsize

    # Save figure
    plt.savefig("../data/graphs/pocet_vozidel_v_cr.png")

    # Show plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
