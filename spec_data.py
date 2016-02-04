import numpy as np
import h5py
from wpca import WPCA


class CleanSpectra(object):
    def __init__(self, min_wavelength=3500, max_wavelength=8300,
                 max_masked_fraction=1.0):
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.max_masked_fraction = max_masked_fraction

    def load_data(self, h5file, selection=None):
        if not isinstance(selection, slice):
            selection = slice(selection)

        self.data = h5py.File(h5file, 'r')
        wavelengths = 10 ** self.data['log_wavelengths'][:]
        mask = ((wavelengths >= self.min_wavelength) &
                (wavelengths <= self.max_wavelength))
        self.wavelengths = wavelengths[mask]
        self.spectra = self.data['spectra'][selection, mask]
        self.weights = self.data['ivars'][selection, mask]

        # remove rows with excessive missing data
        good_rows = (self.weights == 0).mean(1) < self.max_masked_fraction
        self.spectra = self.spectra[good_rows]
        self.weights = self.weights[good_rows] ** 0.5
        return self

    def fit_wpca(self, n_components=200, regularization=False):
        self.wpca = WPCA(n_components=n_components,
                         regularization=regularization)
        self.wpca.fit(self.spectra, weights=self.weights)
        return self

    def reconstruct(self, spectra=None, weights=None, p=2):
        if spectra is None:
            spectra = self.spectra
        if weights is None:
            weights = self.weights

        new_spectra = self.wpca.reconstruct(spectra, weights=weights)
        SN = abs(spectra * weights) ** (1. / p)
        SN /= SN.max(1, keepdims=True)
        return SN * spectra + (1 - SN) * new_spectra


def write_spectra_file(filename, spectra, wavelengths):
    h5f = h5py.File(filename, 'w')

    h5f.create_dataset('spectra', data=spectra)
    h5f.create_dataset('wavelengths', data=wavelengths)
    h5f.close()
