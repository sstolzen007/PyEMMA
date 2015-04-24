# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
Created on 23.01.2015

@author: marscher
'''
import os
import mdtraj
import tempfile
import unittest
import pkg_resources
import numpy as np

from pyemma.coordinates import api
from pyemma.util.log import getLogger

log = getLogger('TestFeatureReader')


def setUpClass(cls):
    # create a fake trajectory which has 3 atoms and coordinates are just a range
    # over all frames.
    cls.output_dir = tempfile.mkdtemp(suffix="test_feature_reader")

    savers = ['.xtc',
              '.trr',
              '.pdb',
              '.dcd',
              '.h5',
              '.binpos',
              #'.nc',
              '.netcdf',
              # '.crd', # broken writer
              # '.mdcrd', # broken writer
              #'.ncdf',
              # '.lh5', # deprecated in mdtraj
              # '.lammpstrj', #
              '.xyz',
              '.gro']

    supported_extensions = savers

    cls.trajfiles = []

    for ext in supported_extensions:
        f = tempfile.mktemp(suffix=ext, dir=cls.output_dir)
        cls.trajfiles.append(f)

    cls.n_frames = 1000

    cls.xyz = np.arange(cls.n_frames * 3 * 3).reshape((cls.n_frames, 3, 3))

    cls.topfile = pkg_resources.resource_filename(
        'pyemma.coordinates.tests.test_featurereader', 'data/test.pdb')
    t = mdtraj.load(cls.topfile)
    t.xyz = cls.xyz
    t.time = np.arange(cls.n_frames)
    for fn in cls.trajfiles:
        t.save(fn)

        # generate test functions for all extension
        _, ext = os.path.splitext(fn)

        def method(self, fn=fn):
            if ext == '.gro':
                pass_top = False
            else:
                pass_top = True
            self._with_lag_and_stride(fn, pass_top)

        name = "test_with_lag_and_stride_" + ext[1:]
        setattr(cls, name, method)

    return cls


class TestFeatureReader(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.output_dir, ignore_errors=True)

    # dummy needed so nose executes this unittest
    def test_dummy(self):
        pass

    def _with_lag_and_stride(self, filename, pass_topology=True):
        if pass_topology:
            reader = api.source(filename, top=self.topfile)
        else:
            reader = api.source()
        strides = [1, 2, 3, 5, 6, 10, 11, 13]
        lags = [1, 2, 7, 11, 23]

        xyz_flattened_2d = self.xyz.reshape((1000, 3 * 3))
        for s in strides:
            for t in lags:
                chunks = []
                chunks_lagged = []
                for _, X, Y in reader.iterator(stride=s, lag=t):
                    chunks.append(X)
                    chunks_lagged.append(Y)

                chunks = np.vstack(chunks)
                chunks_lagged = np.vstack(chunks_lagged)

                np.testing.assert_equal(chunks, xyz_flattened_2d[::s])
                np.testing.assert_equal(chunks, xyz_flattened_2d[t::s])

setUpClass(TestFeatureReader)

if __name__ == "__main__":
    unittest.main()
