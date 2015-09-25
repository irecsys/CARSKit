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
import com.google.common.collect.Table;
import happy.coding.io.Strings;
import librec.data.DenseVector;
import librec.data.DenseMatrix;
import librec.data.SymmMatrix;

/**
 * CAMF: General Class for Context-aware Matrix Factorization (CAMF)
 *
 * @author Yong Zheng
 *
 */

public abstract class CAMF extends ContextRecommender {

    // members for deviation-based models
    protected DenseVector condBias;
    protected DenseMatrix ucBias;
    protected DenseMatrix icBias;

    // members for similarity-based models
    protected SymmMatrix ccMatrix_ICS;
    protected DenseMatrix cfMatrix_LCS;
    protected DenseVector cVector_MCS;


    public CAMF(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {

        super(trainMatrix, testMatrix, fold);

    }

    @Override
    public String toString() {
        return Strings.toString(new Object[]{"numFactors: " + numFactors, "numIter: " + numIters, "lrate: " + initLRate, "maxlrate: " + maxLRate, "regB: " + regB, "regU: " + regU, "regI: " + regI, "regC: " + regC,
                "isBoldDriver: " + isBoldDriver});
    }
}
