import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

u = mda.Universe("/home/debian/water/TIP4P/2005/test/dump_O.lammpstrj", format="LAMMPSDUMP")

ag = u.select_atoms("name O")

rdf = InterRDF(ag, ag, nbins=200, range=(0.0, 10.0))
rdf.run()

plt.plot(rdf.bins, rdf.rdf, label="O-O RDF")
plt.xlabel("Distance (Å)")
plt.ylabel("g(r)")
plt.title("Oxygen-Oxygen Radial Distribution Function")
plt.legend()
plt.grid()
# plt.savefig("rdf_OO.png")
plt.show()
