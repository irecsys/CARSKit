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

package carskit.alg.cars.adaptation.dependent.dev;

import carskit.alg.cars.adaptation.dependent.CAMF;
import carskit.data.setting.Configuration;
import carskit.data.structure.SparseMatrix;
import carskit.generic.ContextRecommender;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import happy.coding.math.Randoms;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;

/**
 * CAMF_C: Baltrunas, Linas, Bernd Ludwig, and Francesco Ricci. "Matrix factorization techniques for context aware recommendation." Proceedings of the fifth ACM conference on Recommender systems. ACM, 2011.
 * <p></p>
 * Note: in this algorithm, there is a rating deviation for each context condition
 *
 * @author Yong Zheng
 *
 */

public class CAMF_C extends CAMF{



    public CAMF_C(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);
        this.algoName = "CAMF_C";
    }

    protected void initModel() throws Exception {

        super.initModel();

        userBias = new DenseVector(numUsers);
        userBias.init(initMean, initStd);

        itemBias = new DenseVector(numItems);
        itemBias.init(initMean, initStd);

        condBias = new DenseVector(numConditions);
        condBias.init(initMean, initStd);

    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        double pred=globalMean + userBias.get(u) + itemBias.get(j) +DenseMatrix.rowMult(P, u, Q, j);
        for(int cond:getConditions(c)){
            pred+=condBias.get(cond);
        }
        return pred;
    }

    @Override
    protected void buildModel() throws Exception {

        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;
            for (MatrixEntry me : trainMatrix) {

                int ui = me.row(); // user-item
                int u= rateDao.getUserIdFromUI(ui);
                int j= rateDao.getItemIdFromUI(ui);
                int ctx = me.column(); // context
                double rujc = me.get();

                double pred = predict(u, j, ctx, false);
                double euj = rujc - pred;

                loss += euj * euj;

                // update factors
                double bu = userBias.get(u);
                double sgd = euj - regB * bu;
                userBias.add(u, lRate * sgd);

                loss += regB * bu * bu;


                double bj = itemBias.get(j);
                sgd = euj - regB * bj;
                itemBias.add(j, lRate * sgd);

                loss += regB * bj * bj;

                double bc_sum=0;
                for(int cond:getConditions(ctx)) {
                    double bc = condBias.get(cond);
                    bc_sum+=bc;
                    sgd = euj - regC * bc;
                    condBias.add(cond, lRate * sgd);
                }

                loss += regB * bc_sum;

                for (int f = 0; f < numFactors; f++) {
                    double puf = P.get(u, f);
                    double qjf = Q.get(j, f);

                    double delta_u = euj * qjf - regU * puf;
                    double delta_j = euj * puf - regI * qjf;

                    P.add(u, f, lRate * delta_u);
                    Q.add(j, f, lRate * delta_j);

                    loss += regU * puf * puf + regI * qjf * qjf;
                }

            }
            loss *= 0.5;

            if (isConverged(iter))
                break;

        }// end of training

    }
}
