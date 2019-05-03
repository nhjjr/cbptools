"""
Configuration file for FSL
"""

from shutil import which
from scipy.sparse import coo_matrix
from shutil import rmtree
import numpy as np
import time
import subprocess
import os


class FSL(object):
    fsl = which('fsl')
    fsl_outputtype = os.getenv('FSLOUTPUTTYPE')
    fsl_dir = os.getenv('FSLDIR')
    probtrackx2 = which('probtrackx2')

    def get_ext(self):
        fsl_extensions = {
            'NIFTI_PAIR': '.hdr',
            'NIFTI_PAIR_GZ': '.hdr.gz',
            'NIFTI': '.nii',
            'NIFTI_GZ': '.nii.gz',
            'ANALYZE': '.hdr',
            'ANALYZE_GZ': '.hdr.gz',
        }
        return fsl_extensions[self.fsl_outputtype]

    def has_fsl(self):
        if self.fsl:
            return True
        else:
            return False

    def has_probtrackx2(self):
        if self.probtrackx2:
            return True
        else:
            return False

    def run_probtrackx2(self, seed: str, target: str, samples: str, mask: str, xfm: str, invxfm: str, tmp_dir: str,
                        options: tuple=None,  wait_for_file: int=240, cleanup_fsl: bool=True) -> np.ndarray:
        """ Run FSL's probtrackx2 function on the input arguments.

        Parameters
        ----------
        seed : str
            Path to the region-of-interest high-resolution mask nifti image. Used for the -x,--seed argument.
        target : str
            Path to the low-resolution target mask nifti image. Used for the --target2 argument.
        samples : str
            Basename for samples files (e.g., 'merged'). Used for the -s,--samples argument.
        mask : str
            Bet binary mask file in diffusion space. Used for the -m,--mask argument.
        xfm : str
            Transform taking seed space to DTI space (either FLIRT matrix of FNIRT warpfield). Used for the --xfm
            argument.
        invxfm : str
            Transform taking DTI space to seed space. Used for the --invxfm argument.
        tmp_dir : str
            Directory to put the FSL final volume output in. Used for the --dir argument. If cleanup_fsl is set to True,
            this directory will be deleted after the connectivity matrix is extracted.
        options : tuple
            Tuple containing further options passed to FSL's probtrackx2 function. Example: ('--nsamples=200')
        wait_for_file: int, optional
            Wait for the FSL output file 'fdt_matrix2' to appear in the file system in seconds, default is 240. If the
            file has not appeared within this time, the script assumes something went wrong.
        cleanup_fsl: bool, optional
            Remove the FSL output directory defined in tmp_dir. Once the connectivity matrix has been extracted, this
            data will no longer be used by the pipeline.

        Returns
        -------
        np.ndarray
            Connectivity matrix derived from FSL's probtrackx2 function

        """
        if not self.has_probtrackx2:
            raise ModuleNotFoundError('fsl: probtrackx2 not found')

        cmd = ('probtrackx2', f'--seed={seed}', f'--target2={target}', f'--samples={samples}', f'--mask={mask}',
               f'--xfm={xfm}', f'--invxfm={invxfm}', f'--dir={tmp_dir}')
        cmd += options
        cmd = ' '.join(cmd)

        # Run FSL's probtrackx2 as a subprocess
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        process.wait()
        fdt_matrix2 = os.path.join(tmp_dir, 'fdt_matrix2.dot')

        # Wait to make sure the file is written to disk
        for i in range(wait_for_file):
            if os.path.exists(fdt_matrix2):
                i, j, value = np.loadtxt(fdt_matrix2, unpack=True)
                i = i.astype(int) - 1  # numpy reads as floats, convert to int for indexing
                j = j.astype(int) - 1  # FSL indexes from 1, but we need 0-indexing

                connectivity = coo_matrix((value, (i, j)))
                connectivity = connectivity.todense(order='C')

                if cleanup_fsl:
                    rmtree(tmp_dir)

                break
            else:
                time.sleep(1)
        else:
            raise FileNotFoundError('fdt_matrix2 was not found on disk. Try increasing the wait_for_file timer')

        return connectivity
