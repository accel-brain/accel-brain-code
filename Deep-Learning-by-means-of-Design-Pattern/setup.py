from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("pydbm/approximation/contrastive_divergence", ["pydbm/approximation/contrastive_divergence.pyx"]),
    Extension("pydbm/synapse/complete_bipartite_graph", ["pydbm/synapse/complete_bipartite_graph.pyx"]),
    Extension("pydbm/synapse_list", ["pydbm/synapse_list.pyx"])
]

setup(
  name = 'pydbm',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

