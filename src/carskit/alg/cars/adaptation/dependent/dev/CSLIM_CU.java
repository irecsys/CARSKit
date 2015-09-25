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

import carskit.data.setting.Configuration;
import carskit.data.structure.DenseVector;
import carskit.data.structure.SparseMatrix;
import carskit.generic.IterativeRecommender;
import carskit.alg.cars.adaptation.dependent.CSLIM;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import happy.coding.io.Lists;
import happy.coding.io.Logs;
import happy.coding.io.Strings;
import librec.data.*;

import java.util.*;
import java.util.Map.Entry;


/**
 * CSLIM_CU: Zheng, Yong, Bamshad Mobasher, and Robin Burke. "CSLIM: Contextual slim recommendation algorithms." Proceedings of the 8th ACM Conference on Recommender Systems. ACM, 2014.
 * <p></p>
 * Note: in this algorithm, there is a rating deviation for each pair of user and context condition; and it is built upon SLIM-I algorithm
 *
 * @author Yong Zheng
 *
 */


@Configuration("binThold, knn, regLw2, regLw1, regLc2, regLc1, similarity, iters, rc")
public class CSLIM_CU extends CSLIM {
    private DenseMatrix W;

    // item's nearest neighbors for kNN > 0
    private Multimap<Integer, Integer> itemNNs;

    // item's nearest neighbors for kNN <=0, i.e., all other items
    private List<Integer> allItems;

    public CSLIM_CU(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isRankingPred = true;
        isCARSRecommender=false; // this option is used to allow the algorithm to call 2D rating matrix "train"
        this.algoName = "CSLIM_CU";


        regLw1 = algoOptions.getFloat("-lw1");
        regLw2 = algoOptions.getFloat("-lw2");
        regLc1 = algoOptions.getFloat("-lc1");
        regLc2 = algoOptions.getFloat("-lc2");

        knn = algoOptions.getInt("-k");
        als = algoOptions.getInt("-als");
    }

    @Override
    protected void initModel() throws Exception {
        super.initModel();

        cuDev = new DenseMatrix(numUsers, numConditions);
        cuDev.init();

        W = new DenseMatrix(numItems, numItems);
        W.init(); // initial guesses: make smaller guesses (e.g., W.init(0.01)) to speed up training


        // find knn based on 2D rating matrix, train
        userCache = train.rowCache(cacheSpec);

        if (knn > 0) {
            // find the nearest neighbors for each item based on item similarity
            SymmMatrix itemCorrs = buildCorrs(false); // this is based on transformed 2D rating matrix, this.train
            itemNNs = HashMultimap.create();

            for (int j = 0; j < numItems; j++) {
                // set diagonal entries to 0
                W.set(j, j, 0);

                // find the k-nearest neighbors for each item
                Map<Integer, Double> nns = itemCorrs.row(j).toMap();

                // sort by values to retriev topN similar items
                if (knn > 0 && knn < nns.size()) {
                    List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nns, true);
                    List<Map.Entry<Integer, Double>> subset = sorted.subList(0, knn);
                    nns.clear();
                    for (Map.Entry<Integer, Double> kv : subset)
                        nns.put(kv.getKey(), kv.getValue());
                }

                // put into the nns multimap
                for (Map.Entry<Integer, Double> en : nns.entrySet())
                    itemNNs.put(j, en.getKey());
            }
        } else {
            // all items are used
            allItems = train.columns();

            for (int j = 0; j < numItems; j++)
                W.set(j, j, 0.0);
        }
    }

    @Override
    protected void buildModel() throws Exception {


        // number of iteration cycles
        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;

            for (MatrixEntry me : trainMatrix) {

                int ui = me.row(); // user-item
                int u= rateDao.getUserIdFromUI(ui);
                int j= rateDao.getItemIdFromUI(ui);
                int c = me.column(); // context
                double rujc = me.get();

                double pred = predict(u, j, c, true, j);
                double eujc = rujc - pred;
                loss += eujc * eujc;

                // find k-nearest neighbors
                Collection<Integer> nns = knn > 0 ? itemNNs.get(j) : allItems;

                // update factors

                Collection<Integer> conditions=rateDao.getContextConditionsList().get(c);
                double dev_c=0;
                for(Integer cond:conditions)
                {
                    dev_c+=cuDev.get(u,cond);
                }
                SparseVector Ru = userCache.get(u);

                // start updating W
                double sum_w=0;
                for (int k : nns) {
                    double update=W.get(k, j);
                    sum_w += update;

                    loss += regLw2*update*update + regLw1*update;

                    double delta_w = eujc*(Ru.get(k) + dev_c) - regLw2*update - regLw1;
                    update += lRate*delta_w;
                    W.set(k,j,update);
                }

                // start updating cuDev
                for(Integer cond:conditions)
                {
                    double update = cuDev.get(u,cond);

                    loss += regLc2*update*update + regLc1*update;

                    double delta_c = eujc*sum_w - regLc2*update - regLc1;
                    update += lRate*delta_c;
                    cuDev.set(u, cond, update);
                }// end train


            }



        }
    }

    protected double predict(int u, int j, int c, boolean exclude, int excluded_item) throws Exception {


            Collection<Integer> nns = knn > 0 ? itemNNs.get(j) : allItems;
            SparseVector Ru = userCache.get(u);

            Collection<Integer> conditions=rateDao.getContextConditionsList().get(c);
            double dev_c=0;
            for(Integer cond:conditions)
            {
                dev_c+=cuDev.get(u,cond);
            }

            double pred = 0;
            for (int k : nns) {
                if (Ru.contains(k)) {
                    if(exclude==true && k == excluded_item)
                        continue;
                    else {
                        double ruk = Ru.get(k);
                        pred += (ruk + dev_c) * W.get(k, j);
                    }
                }
            }

            return pred;

    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        return predict(u,j,c,true,j);
    }

    @Override
    protected boolean isConverged(int iter) {
        double delta_loss = last_loss - loss;
        last_loss = loss;

        if (verbose)
            Logs.debug("{}{} iter {}: loss = {}, delta_loss = {}", algoName, foldInfo, iter, loss, delta_loss);

        return iter > 1 ? delta_loss < 1e-5 : false;
    }


}

