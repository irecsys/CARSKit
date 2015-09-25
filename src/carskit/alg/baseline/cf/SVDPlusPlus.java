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
import librec.data.MatrixEntry;

import carskit.data.structure.SparseMatrix;
import java.util.List;

/**
 * SVD++: Koren, Yehuda, Robert Bell, and Chris Volinsky. "Matrix factorization techniques for recommender systems." Computer 8 (2009): 30-37.<p></p>
 * Note: This implementation is modified from the algorithm in LibRec-v1.3
 *
 * @author Yong Zheng
 *
 */

public class SVDPlusPlus extends BiasedMF {

    protected DenseMatrix Y;

    public SVDPlusPlus(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        algoName = "SVD++";
    }

    @Override
    protected void initModel() throws Exception {
        super.initModel();

        Y = new DenseMatrix(numItems, numFactors);
        Y.init(initMean, initStd);

        userItemsCache = train.rowColumnsCache(cacheSpec);
    }

    @Override
    protected void buildModel() throws Exception {

        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;
            for (MatrixEntry me : train) {

                int u = me.row(); // user
                int j = me.column(); // item
                double ruj = me.get();

                double pred = predict(u, j);
                double euj = ruj - pred;

                loss += euj * euj;

                List<Integer> items = userItemsCache.get(u);
                double w = Math.sqrt(items.size());

                // update factors
                double bu = userBias.get(u);
                double sgd = euj - regB * bu;
                userBias.add(u, lRate * sgd);

                loss += regB * bu * bu;

                double bj = itemBias.get(j);
                sgd = euj - regB * bj;
                itemBias.add(j, lRate * sgd);

                loss += regB * bj * bj;

                double[] sum_ys = new double[numFactors];
                for (int f = 0; f < numFactors; f++) {
                    double sum_f = 0;
                    for (int k : items)
                        sum_f += Y.get(k, f);

                    sum_ys[f] = w > 0 ? sum_f / w : sum_f;
                }

                for (int f = 0; f < numFactors; f++) {
                    double puf = P.get(u, f);
                    double qjf = Q.get(j, f);

                    double sgd_u = euj * qjf - regU * puf;
                    double sgd_j = euj * (puf + sum_ys[f]) - regI * qjf;

                    P.add(u, f, lRate * sgd_u);
                    Q.add(j, f, lRate * sgd_j);

                    loss += regU * puf * puf + regI * qjf * qjf;

                    for (int k : items) {
                        double ykf = Y.get(k, f);
                        double delta_y = euj * qjf / w - regU * ykf;
                        Y.add(k, f, lRate * delta_y);

                        loss += regU * ykf * ykf;
                    }
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
    protected double predict(int u, int j) throws Exception {

        double pred = globalMean + userBias.get(u) + itemBias.get(j) + DenseMatrix.rowMult(P, u, Q, j);
        List<Integer> items = userItemsCache.get(u);
        double w = Math.sqrt(items.size());
        for (int k : items)
            pred += DenseMatrix.rowMult(Y, k, Q, j) / w;
        return pred;
    }
}
