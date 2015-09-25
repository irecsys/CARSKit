// Copyright (C) 2015 Yong Zheng
//
// This file is part of CARSKit.
//
// CARSKit is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CARSKit is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CARSKit. If not, see <http://www.gnu.org/licenses/>.
//

package carskit.alg.cars.adaptation.dependent;

import carskit.data.structure.SparseMatrix;
import carskit.generic.ContextRecommender;
import carskit.generic.IterativeRecommender;
import happy.coding.io.Strings;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SymmMatrix;

/**
 * CSLIM: General Class for Contextual Sparse Linear Method (CSLIM)
 *
 * @author Yong Zheng
 *
 */


public abstract class CSLIM extends ContextRecommender{

    // members for deviation-based models
    protected DenseVector cDev;
    protected DenseMatrix ciDev;
    protected DenseMatrix cuDev;
    protected DenseMatrix ccDev;

    // members for similarity-based models
    protected SymmMatrix ccMatrix_ICS;
    protected DenseMatrix cfMatrix_LCS;
    protected DenseVector cVector_MCS;

    // regularization parameters for the L1 or L2 term
    protected float regLw1, regLw2, regLc1, regLc2;
    protected int als;

    protected double lowbound;

    public CSLIM(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);
        lowbound = 1.0/Math.pow(10, 100);
    }

    @Override
    public String toString() {
        return Strings.toString(new Object[]{"numFactors: " + numFactors, "numIter: " + numIters, "lrate: " + initLRate, "maxlrate: " + maxLRate, "regB: " + regB, "regU: " + regU, "regI: " + regI, "regC: " + regC,
                "knn: "+knn, "regLw1: "+regLw1, "regLw2: "+regLw2, "regLc1: "+regLc1, "regLc2: "+regLc2, "isBoldDriver: " + isBoldDriver});
    }
}
