import os
import gc
import argparse

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

    if Nmax is None:
        Nmax_disp = 'all'
    else:
        Nmax_disp = Nmax

    with timeit("loading {0} spectra from {1}".format(Nmax_disp, input_file)):
        cln.load_data(input_file, Nmax)

    with timeit("fitting weighted PCA for {0} components"
                "".format(n_components)):
        cln.fit_wpca(n_components=n_components)

    with timeit("computing {0} reconstructed spectra:"
                "".format(cln.spectra.shape[0])):
        new_spectra = cln.reconstruct(p=p)

    # clean up memory
    wavelengths = cln.wavelengths.copy()
    del cln

    with timeit("writing reconstructed spectra to file"):
        write_spectra_file(output_file,
                           spectra=new_spectra,
                           wavelengths=wavelengths)

def main():
    parser = argparse.ArgumentParser(description='Clean spectra files')
    parser.add_argument('filenames', type=str, nargs='+')
    args = parser.parse_args()

    for filename in args.filenames:
        if not os.path.exists(filename) or not filename.endswith('.hdf5'):
            raise ValueError("{0} is not a valid HDF5 file".format(filename))

    for filename in args.filenames:
        base, ext = os.path.splitext(filename)
        output_file = base + "_clean" + ext
        print("\n============================================================")
        print("Cleaning", filename, "->", output_file)
        process_file(input_file=filename,
                     output_file=output_file)
        gc.collect()

if __name__ == '__main__':
    main()
