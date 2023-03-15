"""
Simple script to perform PSF photometry on a folder with FITS files in
parallel.
"""

# Standard library
import sys
import logging
from warnings import filterwarnings
from contextlib import contextmanager
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable
from time import perf_counter

# 3rth party
import numpy as np
import pandas as pd

from astropy.stats import gaussian_sigma_to_fwhm
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.io import fits

from photutils.detection import DAOStarFinder
from photutils.psf import DAOGroup
from photutils.psf import IntegratedGaussianPRF
from photutils.background import MMMBackground
from photutils.psf import IterativelySubtractedPSFPhotometry

# Configs
logging.basicConfig(
        level="DEBUG",
        format="%(asctime)s %(levelname)s -> %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
)

# Remove this warning because it's expected and will aways happen
filterwarnings("ignore", message=r".*The fit may be unsuccessful.*")

@contextmanager
def timing(label: str = ""):
    """
    Timing co routine. Logs to the main logger the time of execution for the
    code within the with statement.
    """
    tic = perf_counter()
    yield
    toc = perf_counter()
    
    ellapsed_time = toc - tic
    unit = "sec"

    if ellapsed_time > 60:
        ellapsed_time /= 60
        unit = "min"

    logging.info(f"Finished processing: {label} : ellapsed time {ellapsed_time:.2f} {unit}")
        

def to_mag(flux: float, zero_point: float = 25.):
    """Given a flux and a zero point converts flux to magnitude"""
    return zero_point - 2.512*np.log10(flux)


class PhotSubprocessWrapper:
    """Wraps psf_phot call to tackle subprocess details"""

    def __init__(self, phot_func: Callable, out_path: Path, niters: int = 3) -> None:
        self.phot_func = phot_func
        self.out_path = out_path
        self.niters = niters


    def __call__(self, args: tuple[Path, float, float]):
        """Target function to be run by a subprocess"""
        image_path, fwhm, sky_sigma = args  # unpack parameters

        # Check if calculation was already done
        out = self.out_path / f"phot_{image_path.name.split('.')[0]}.csv"

        if out.exists():
            logging.info(f"PSF for { image_path } alredy done. Returning None.") 

            return None


        # Log stuff
        logging.info(f"Running PSF photometry: {image_path = }, {fwhm = :.2f}, {sky_sigma = :.2f}") # make logging

        with timing(f"PSF: {image_path.name}"):
            rv = self.phot_func(image_path, fwhm, sky_sigma, niters=self.niters) # run actual function

        rv.to_csv(out)

        return out


def psf_phot(image_path: Path, fwhm: float, sky_sigma: float, niters: int = 3) -> pd.DataFrame:
    """
    Given an `image` as a np.array, it's `fwhm` and the `sky_sigma` as floats
    performs the PSF photometry of the sources on the `image` and returns the
    resulting catalog.


    Parameters
    ----------
        image_path: pathlib.Path
            Relative path to the desired FITS image.

        fwhm: float
            Full with half maxima representative of the image.

        sky_sigma: float
            Spread estimative of the sky background.

    Returns
    -------
        result: pd.DataFrame
            Dataframe containing the photometry.
    """

    image = load_fits(image_path)
    
    finder = DAOStarFinder(threshold = 3.5*sky_sigma, fwhm=fwhm)
    grouper = DAOGroup(2*fwhm)
    mmm_bkg = MMMBackground()
    psf_model = IntegratedGaussianPRF(sigma=fwhm/gaussian_sigma_to_fwhm)
    
    phot = IterativelySubtractedPSFPhotometry(
            finder=finder,
            group_maker=grouper,
            bkg_estimator=mmm_bkg,
            psf_model=psf_model,
            fitter=LevMarLSQFitter(),
            niters=niters, fitshape=(11,11))

    result = phot(image=image).to_pandas()
    result["magnitude"] = to_mag(result["flux_fit"])
    
    return result


def load_fits(path: Path) -> np.ndarray:
    """Given a `path` loads the image as a 2D numpy array"""
    image: np.ndarray = fits.getdata(path)
    return image

def parse_args():

    parser = ArgumentParser(
            prog="psf_phot",
            description="Performs PSF photometry in a folder of FITS images"
    )

    parser.add_argument(
            "folder",
            type=Path,
            help="Folder with the FITS files"
    )

    parser.add_argument(
            "out_folder",
            type=Path,
            help="Folder to put resulting catalogs"
    )

    parser.add_argument(
            "info_file", 
            type=Path,
            help="File with table w/ parameters for photometry (generated with wdp.inspect)"
    )

    parser.add_argument(
            "-w", "--workers", 
            type=int,
            help="Number of worker processes, if -1, launches (n_cores-1) worker processes",
            default=-1,
            choices=range(-1, cpu_count() + 1),
            metavar=f"[-1, 1 .. {cpu_count()}]"
    )

    parser.add_argument(
            "-n", "--niters",
            type=int, 
            help="Number of passages of the PSF",
            default=2
    )

    parser.add_argument(
            "-v", "--verbose",
            action="store_true",
    )

    return parser.parse_args()


def main():
    """
    Reads paths from the shell for a folder containing FITS files and a csv
    file containing information about the field, to perform PSF photometry on
    the images on the folder using the information from the inspection file.


    Inputs
    ------
        folder_path: str
            Path to folder with the fits files

        info_path: str
            Path to the csv file containing the information necessary for the 
            PSF photometry ("FWHM" and "sky_sigma")

    Side-Effects
    ------------
        Creates out folder and writes a csv file with PSF photometry table for
        each FITS file
    """
    args = parse_args()
    
    folder: Path = args.folder
    out_folder: Path = args.out_folder
    info_file: Path = args.info_file
    niters: int = args.niters
    n_workers: int = args.workers

    if n_workers == -1:
        n_workers = cpu_count() - 1

    if not out_folder.exists():
        out_folder.mkdir(parents=True)

    logging.info(f"Starting pipeline: {folder = }, {info_file = }, {n_workers = }, {niters = }")

    # Load info file
    info = pd.read_csv(info_file)
    logging.debug(info)

    # Get fits files paths
    fits_files = folder.glob("*.fits")

    # Build tuple containg necessary parameters to psf_phot
    arguments = []
    for f in sorted(fits_files):

        try:
            logging.debug(f)
            line = info.query("file == @f.name").iloc[0]
            args = (f, line.FWHM, line.sky_sigma)
            arguments.append(args)
            logging.debug(args)

        except IndexError:
            logging.warning(f"{f.name} doesn't have infomation available")

    # Start multiprocessing pool
    with Pool(processes=n_workers) as worker_pool:

        target = PhotSubprocessWrapper(psf_phot, out_folder, niters=niters)
        results = worker_pool.imap(target, arguments)

        for res in results:
            logging.info(f"Processed image and saved result: {res}")


if __name__ == "__main__":

    with timing("Main loop"):
            main()
