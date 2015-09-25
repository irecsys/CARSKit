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

import happy.coding.io.Logs;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import carskit.data.structure.SparseMatrix;
import carskit.generic.IterativeRecommender;

/**
 * BiasedMF: Koren, Yehuda, Robert Bell, and Chris Volinsky. "Matrix factorization techniques for recommender systems." Computer 8 (2009): 30-37.<p></p>
 * Note: This implementation is modified from the algorithm in LibRec-v1.3
 *
 * @author Yong Zheng
 *
 */


public class BiasedMF extends IterativeRecommender {

    public BiasedMF(SparseMatrix rm, SparseMatrix tm, int fold) {

        super(rm, tm, fold);
        this.algoName = "BiasedMF";
    }

    protected void initModel() throws Exception {

        super.initModel();

        userBias = new DenseVector(numUsers);
        itemBias = new DenseVector(numItems);

        // initialize user bias
        userBias.init(initMean, initStd);
        itemBias.init(initMean, initStd);
    }

    @Override
    protected void buildModel() throws Exception {

        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;
            for (MatrixEntry me : train) {

                int u = me.row(); // user
                int j = me.column(); // item
                double ruj = me.get();

                double pred = predict(u, j, -1, false);
                double euj = ruj - pred;

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


    @Override
    protected double predict(int u, int j) throws Exception {
        return globalMean + userBias.get(u) + itemBias.get(j) + DenseMatrix.rowMult(P, u, Q, j);
    }

    public double getMaxUserID()
    {
        int uid=-1;
        for(int u:rateDao.getUserIds().inverse().keySet())
            if(u>uid)
                uid=u;
        return uid;
    }

    public double getMaxItemID()
    {
        int uid=-1;
        for(int u:rateDao.getItemIds().inverse().keySet())
            if(u>uid)
                uid=u;
        return uid;
    }

}
