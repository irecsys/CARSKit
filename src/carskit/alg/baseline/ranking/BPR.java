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

package carskit.alg.baseline.ranking;

import happy.coding.io.Strings;
import happy.coding.math.Randoms;
import librec.data.SparseVector;

import carskit.data.structure.SparseMatrix;
import carskit.generic.IterativeRecommender;

/**
 * BPR: Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence. AUAI Press, 2009.
 * <p></p>
 * Note: This implementation is modified from the algorithm in LibRec-v1.3
 *
 * @author Yong Zheng
 *
 */

public class BPR extends IterativeRecommender {

    public BPR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isRankingPred = true;
        initByNorm = false;
        this.algoName = "BPR";
    }

    @Override
    protected void initModel() throws Exception {
        super.initModel();

        userCache = train.rowCache(cacheSpec);
    }

    @Override
    protected void buildModel() throws Exception {

        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;
            for (int s = 0, smax = numUsers * 100; s < smax; s++) {

                // randomly draw (u, i, j)
                int u = 0, i = 0, j = 0;

                while (true) {
                    u = Randoms.uniform(numUsers);
                    SparseVector pu = userCache.get(u);

                    if (pu.getCount() == 0)
                        continue;

                    int[] is = pu.getIndex();
                    i = is[Randoms.uniform(is.length)];

                    do {
                        j = Randoms.uniform(numItems);
                    } while (pu.contains(j));

                    break;
                }

                // update parameters
                double xui = predict(u, i);
                double xuj = predict(u, j);
                double xuij = xui - xuj;

                double vals = -Math.log(g(xuij));
                loss += vals;

                double cmg = g(-xuij);

                for (int f = 0; f < numFactors; f++) {
                    double puf = P.get(u, f);
                    double qif = Q.get(i, f);
                    double qjf = Q.get(j, f);

                    P.add(u, f, lRate * (cmg * (qif - qjf) - regU * puf));
                    Q.add(i, f, lRate * (cmg * puf - regI * qif));
                    Q.add(j, f, lRate * (cmg * (-puf) - regI * qjf));

                    loss += regU * puf * puf + regI * qif * qif + regI * qjf * qjf;
                }
            }

            if (isConverged(iter))
                break;

        }
    }

    @Override
    public String toString() {
        return Strings.toString(new Object[] { binThold, numFactors, initLRate, maxLRate, regU, regI, numIters }, ",");
    }
}