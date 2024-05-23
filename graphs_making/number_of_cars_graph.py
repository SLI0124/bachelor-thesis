import matplotlib.pyplot as plt

# Data
# https://www.cappo.cz/cisla-a-fakta/stav-vozoveho-parku-v-cr
# https://portal.sda-cia.cz/clanek.php?id=243
years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022']
# population = [5_368_661, 5_592_738, 5_802_521, 5_989_538, 6_129_874, 6_293_125, 6_425_417]
population_cz = [5.368, 5.592, 5.802, 5.989, 6.129, 6.293, 6.425]


def main():
    plt.figure(figsize=(16, 12))
    plt.bar(years, population_cz, color='skyblue')

    # Labeling
    plt.xlabel('Rok', fontsize=28)
    plt.ylabel('Počet osobních vozidel [v miliónech] ', fontsize=24)
    plt.title('Počet osobních vozidel v České Republice', fontsize=28)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.ylim(5, 6.5)

    for i, txt in enumerate(population_cz):
        plt.annotate(txt, (years[i], population_cz[i]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center',
                     fontsize=24)

    # Save figure
    plt.savefig("../data/graphs/pocet_vozidel_v_cr.png")

    # Show plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
