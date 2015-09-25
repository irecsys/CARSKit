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

package carskit.alg.baseline.cf;

import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import carskit.data.structure.SparseMatrix;
import carskit.generic.IterativeRecommender;

import java.util.Iterator;

/**
 * PMF: Mnih, Andriy, and Ruslan Salakhutdinov. "Probabilistic matrix factorization." Advances in neural information processing systems. 2007.
 * <p></p>
 * Note: This implementation is modified from the algorithm in LibRec-v1.3
 *
 * @author Yong Zheng
 *
 */

public class PMF extends IterativeRecommender {

    public PMF(SparseMatrix rm, SparseMatrix tm, int fold) {

        super(rm, tm, fold);
        this.algoName = "PMF";
    }

    @Override
    protected void buildModel() throws Exception {

        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;
            for (MatrixEntry me : train) {

                int u = me.row(); // user
                int j = me.column(); // item
                double ruj = me.get();

                double puj = predict(u, j, -1, false);
                double euj = ruj - puj;

                loss += euj * euj;

                // update factors
                for (int f = 0; f < numFactors; f++) {
                    double puf = P.get(u, f), qjf = Q.get(j, f);

                    P.add(u, f, lRate * (euj * qjf - regU * puf));
                    Q.add(j, f, lRate * (euj * puf - regI * qjf));

                    loss += regU * puf * puf + regI * qjf * qjf;
                }

            }

            loss *= 0.5;

            if (isConverged(iter))
                break;

        }// end of training

    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        if(isUserSplitting)
            u = userIdMapper.contains(u,c) ? userIdMapper.get(u,c) : u;
        if(isItemSplitting)
            j = itemIdMapper.contains(j,c) ? itemIdMapper.get(j,c) : j;
        return predict(u,j);
    }

    @Override
    protected double predict(int u, int j) throws Exception{

        return DenseMatrix.rowMult(P, u, Q, j);
    }
}
