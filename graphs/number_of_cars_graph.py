import matplotlib.pyplot as plt

years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
population = [4496232, 4581642, 4706325, 4729185, 4833386, 5115316, 5277619, 5435273, 5584743, 5735796, 5882916, 6024463, 6162433, 6296786]

plt.figure(figsize=(16, 9))
plt.bar(years, population, color='skyblue', width=0.8)

plt.xlabel('Rok', fontsize=18)
plt.ylabel('Počet osobních vozidel [v miliónech] ', fontsize=18)
plt.title('Počet osobních vozidel v České Republice', fontsize=22)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

for i, txt in enumerate(population):
    plt.annotate(txt, (years[i], population[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12)  # Increase the font size


plt.savefig("pocet_vozidel_v_cr.png")


plt.tight_layout()
plt.show()
