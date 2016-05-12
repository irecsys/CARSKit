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


import carskit.generic.Recommender;
import librec.data.SparseMatrix;
import librec.data.DenseVector;
import librec.data.SparseVector;
import librec.data.SymmMatrix;

import java.util.Map;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import happy.coding.io.Lists;
import happy.coding.io.Strings;
import happy.coding.math.Stats;
import happy.coding.io.Logs;

/**
 * UserKNN: Resnick, Paul, et al. "GroupLens: an open architecture for collaborative filtering of netnews." Proceedings of the 1994 ACM conference on Computer supported cooperative work. ACM, 1994.
 * <p></p>
 * Note: This implementation is modified from the algorithm in LibRec-v1.3
 *
 * @author Yong Zheng
 *
 */

public class UserKNN extends Recommender {

    // user: nearest neighborhood
    private SymmMatrix userCorrs;
    private DenseVector userMeans;


    public UserKNN(carskit.data.structure.SparseMatrix trainMatrix, carskit.data.structure.SparseMatrix testMatrix, int fold) {

        super(trainMatrix, testMatrix, fold);
        this.algoName = "UserKNN";

    }


    @Override
    protected void initModel() throws Exception {
        super.initModel();
        userCorrs = buildCorrs(true);
        userMeans = new DenseVector(numUsers);
        for (int u = 0; u < numUsers; u++) {
            SparseVector uv = train.row(u);
            userMeans.set(u, uv.getCount() > 0 ? uv.mean() : globalMean);
        }
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
        // find a number of similar users
        Map<Integer, Double> nns = new HashMap<>();

        SparseVector dv = userCorrs.row(u);
        for (int v : dv.getIndex()) {
            double sim = dv.get(v);
            double rate = train.get(v, j);

            if (isRankingPred && rate > 0)
                nns.put(v, sim); // similarity could be negative for item ranking
            else if (sim > 0 && rate > 0)
                nns.put(v, sim);
        }

        // topN similar users
        if (knn > 0 && knn < nns.size()) {
            List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nns, true);
            List<Map.Entry<Integer, Double>> subset = sorted.subList(0, knn);
            nns.clear();
            for (Map.Entry<Integer, Double> kv : subset)
                nns.put(kv.getKey(), kv.getValue());
        }

        if (nns.size() == 0)
            return  globalMean;
        else {
            double sum = 0, ws = 0;
            for (Entry<Integer, Double> en : nns.entrySet()) {
                int v = en.getKey();
                double sim = en.getValue();
                double rate = train.get(v, j);

                sum += sim * (rate - userMeans.get(v));
                ws += Math.abs(sim);
            }

            return ws > 0 ? userMeans.get(u) + sum / ws : globalMean;
        }

    }

    @Override
    public String toString() {
        return Strings.toString(new Object[] { knn, similarityMeasure, similarityShrinkage });
    }

}
