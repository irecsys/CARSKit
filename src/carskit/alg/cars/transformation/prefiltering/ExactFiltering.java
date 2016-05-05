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

package carskit.alg.cars.transformation.prefiltering;

import carskit.generic.IterativeRecommender;
import carskit.generic.Recommender;
import happy.coding.io.Lists;
import happy.coding.io.Logs;
import happy.coding.io.Strings;
import librec.data.*;

import java.util.*;

/**
 * Note: this algorithm is built upon UserKNN
 *
 * @author Yong Zheng
 *
 */

public class ExactFiltering extends Recommender {
    // user: nearest neighborhood
    private SymmMatrix userCorrs;
    private DenseVector userMeans;


    public ExactFiltering(carskit.data.structure.SparseMatrix trainMatrix, carskit.data.structure.SparseMatrix testMatrix, int fold) {

        super(trainMatrix, testMatrix, fold);
        this.algoName = "ExactFiltering";
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
    protected double predict(int a, int t, int c) throws Exception {
        double pred = 0;

        double part3=0, part3_count=0;
        HashMap<Integer, Double> part22 =new  HashMap<Integer, Double>();
        HashMap<Integer, Double> part22_count =new  HashMap<Integer, Double>();
        HashMap<Integer, Double> part21 = new HashMap<>();

        Set<Integer> users= rateDao.getUserList(trainMatrix);
        HashMap<Integer, Double> nns = new HashMap<>();
        HashMap<Integer, Double> nns_ratings = new HashMap<>();

        for(int u:users){
            if(u!=a){
                int ui = rateDao.getUserItemId(u+","+t);
                if(ui!=-1)
                {
                    double rate = trainMatrix.get(ui, c);
                    if(rate>0){
                        // this user has rated item t in context c
                        double sim = userCorrs.get(a, u);
                        if(sim>0) {
                            nns.put(u,sim);
                            nns_ratings.put(u, rate);
                        }
                    }
                }
            }
        }

        // start calculations
        // top-N neighbors
        List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nns, true);
        int k = nns.size();
        if(k!=0) {
            k = (k > knn) ? knn : k;
            List<Map.Entry<Integer, Double>> subset = sorted.subList(0, k);
            nns.clear();
            for (Map.Entry<Integer, Double> kv : subset)
                nns.put(kv.getKey(), kv.getValue());

            double sum1 = 0;
            double sum2 = 0;

            for (Map.Entry<Integer, Double> en : nns.entrySet()) {
                int ngbr = en.getKey();
                sum2 += en.getValue();
                sum1 += (nns_ratings.get(ngbr) - userMeans.get(ngbr))*en.getValue();
            }

            pred = userMeans.get(a) + sum1/sum2;
        }

        return (pred>0)?pred:userMeans.get(a);
    }

    @Override
    public String toString() {
        return Strings.toString(new Object[] { knn, similarityMeasure, similarityShrinkage });
    }

}
