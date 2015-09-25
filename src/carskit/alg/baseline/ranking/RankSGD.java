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



import happy.coding.io.Lists;
import happy.coding.io.Strings;
import happy.coding.math.Randoms;

import java.util.HashMap;
import java.util.List;
import java.util.Map;


import librec.data.SparseVector;
import librec.data.VectorEntry;


import carskit.data.structure.SparseMatrix;
import carskit.generic.IterativeRecommender;

/**
 * RankSGD: Jahrer, Michael, and Andreas Töscher. "Collaborative Filtering Ensemble." KDD Cup. 2012.
 * <p></p>
 * Note: This implementation is modified from the algorithm in LibRec-v1.3
 *
 * @author Yong Zheng
 *
 */

public class RankSGD extends IterativeRecommender {

    // item sampling probabilities sorted ascendingly
    protected List<Map.Entry<Integer, Double>> itemProbs;

    public RankSGD(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isRankingPred = true;
        this.algoName = "RankSGD";
        checkBinary();
    }

    @Override
    protected void initModel() throws Exception {
        super.initModel();

        // compute item sampling probability
        Map<Integer, Double> itemProbsMap = new HashMap<>();
        for (int j = 0; j < numItems; j++) {
            int users = train.columnSize(j);

            // sample items based on popularity
            double prob = (users + 0.0) / numRates;
            if (prob > 0)
                itemProbsMap.put(j, prob);
        }
        itemProbs = Lists.sortMap(itemProbsMap);
    }

    @Override
    protected void buildModel() throws Exception {
        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;

            // for each rated user-item (u,i) pair
            for (int u : train.rows()) {

                SparseVector Ru = train.row(u);
                for (VectorEntry ve : Ru) {
                    // each rated item i
                    int i = ve.index();
                    double rui = ve.get();

                    int j = -1;
                    while (true) {
                        // draw an item j with probability proportional to
                        // popularity
                        double sum = 0, rand = Randoms.random();
                        for (Map.Entry<Integer, Double> en : itemProbs) {
                            int k = en.getKey();
                            double prob = en.getValue();

                            sum += prob;
                            if (sum >= rand) {
                                j = k;
                                break;
                            }
                        }

                        // ensure that it is unrated by user u
                        if (!Ru.contains(j))
                            break;
                    }
                    double ruj = 0;

                    // compute predictions
                    double pui = predict(u, i), puj = predict(u, j);

                    double e = (pui - puj) - (rui - ruj);
                    loss += e * e;

                    // update vectors
                    double ye = lRate * e;
                    for (int f = 0; f < numFactors; f++) {
                        double puf = P.get(u, f);
                        double qif = Q.get(i, f);
                        double qjf = Q.get(j, f);

                        P.add(u, f, -ye * (qif - qjf));
                        Q.add(i, f, -ye * puf);
                        Q.add(j, f, ye * puf);

                    }
                }
            }

            loss *= 0.5;

            if (isConverged(iter))
                break;
        }
    }

    @Override
    public String toString() {
        return Strings.toString(new Object[] { binThold, initLRate, numIters }, ",");
    }
}

