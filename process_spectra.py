from time import time
from contextlib import contextmanager

from spec_data import CleanSpectra, write_spectra_file


@contextmanager
def timeit(message):
    if message:
        print(message)
    t = time()
    yield
    print("   {0:.2g} sec".format(time() - t))


def process_file(input_file, output_file, Nmax=None,
                 n_components=200, p=2):
    cln = CleanSpectra()

    with timeit("loading {0} spectra from {1}".format(Nmax, input_file)):
        cln.load_data(input_file, Nmax)

    with timeit("fitting weighted PCA for {0} components"
                "".format(n_components)):
        cln.fit_wpca(n_components=n_components)

    with timeit("computing {0} reconstructed spectra:"
                "".format(cln.spectra.shape[0])):
        new_spectra = cln.reconstruct(p=p)

    with timeit("writing reconstructed spectra to file"):
        write_spectra_file('spectra100000_clean.hdf5',
                       spectra=new_spectra,
                       wavelengths=cln.wavelengths)


process_file(input_file='spectra100000.hdf5',
             output_file='spectra100000_clean.hdf5')
