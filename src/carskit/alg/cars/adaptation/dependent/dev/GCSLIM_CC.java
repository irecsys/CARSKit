
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
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import happy.coding.io.Lists;
import happy.coding.io.Logs;
import happy.coding.io.Strings;
import happy.coding.math.Stats;
import librec.data.*;

import java.util.*;
import java.util.Map.Entry;


/**
 * GCSLIM_CC: Zheng, Yong, Bamshad Mobasher, and Robin Burke. "Deviation-Based Contextual SLIM Recommenders." Proceedings of the 23rd ACM International Conference on Conference on Information and Knowledge Management. ACM, 2014.
 * <p></p>
 * Note: in this algorithm, there is a rating deviation between each two context conditions; it is built upon SLIM-I algorithm
 *
 * @author Yong Zheng
 *
 */


@Configuration("binThold, knn, regLw2, regLw1, regLc2, regLc1, similarity, iters, rc")
public class GCSLIM_CC extends CSLIM {
    private DenseMatrix W;

    // item's nearest neighbors for kNN > 0
    private Multimap<Integer, Integer> itemNNs;

    // item's nearest neighbors for kNN <=0, i.e., all other items
    private List<Integer> allItems;

    public GCSLIM_CC(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isRankingPred = true;
        isCARSRecommender=false; // this option is used to allow the algorithm to call 2D rating matrix "train"
        this.algoName = "GCSLIM_CC";

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

        ccDev=new DenseMatrix(numConditions, numConditions);
        ccDev.init();
        for(int i=0;i<numConditions;++i)
            ccDev.set(i,i,0.0);

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

                Collection<Integer> nns = knn > 0 ? itemNNs.get(j) : allItems;
                SparseVector Ru = userCache.get(u);

                HashBasedTable<Integer, Integer, Double> Dev_weights=HashBasedTable.create();
                HashBasedTable<Integer, Integer, Double>  Weight_devs=HashBasedTable.create();

                double pred = 0;
                for (int k : nns) {
                    if (Ru.contains(k)) {
                        if(k != j) {
                            // extract a random contextual rating by user u and item k
                            String key=u+","+k;
                            int uiid=rateDao.getUserItemId(key);
                            List<Integer> ctxid=this.trainMatrix.getColumns(uiid);

                            Random r = new Random();
                            int index = r.nextInt(ctxid.size());
                            int ctx=ctxid.get(index);

                            // get rating for u, k, ctx
                            double ruk = this.trainMatrix.get(uiid, ctx);

                            String[] sfrom=rateDao.getContextId(ctx).split(",");
                            String[] sto=rateDao.getContextId(c).split(",");
                            double dev_c=0;
                            double w = W.get(k, j);
                            for(int i=0;i<sfrom.length;++i){
                                int cond1=Integer.valueOf(sfrom[i]);
                                int cond2=Integer.valueOf(sto[i]);
                                dev_c+=ccDev.get(cond1,cond2);
                                if(cond1!=cond2) {
                                    if (Dev_weights.contains(cond1, cond2))
                                        Dev_weights.put(cond1, cond2, w + Dev_weights.get(cond1, cond2));
                                    else
                                        Dev_weights.put(cond1, cond2, w);
                                }
                            }
                            Weight_devs.put(k, j, dev_c + ruk);
                            pred += (ruk + dev_c) * w;

                        }

                    }
                }

                double eujc = rujc - pred;
                loss += eujc * eujc;

                for(int idk:Weight_devs.rowKeySet())
                    for(int idj:Weight_devs.row(idk).keySet()){

                        double update=W.get(idk,idj);

                        loss += regLw2*update*update + regLw1*update;


                        double delta_w = eujc*Weight_devs.get(idk,idj) - regLw2*update - regLw1;
                        update += lRate*delta_w;
                        W.set(idk,idj,update);

                    }

                // start updating cDev
                for(int cond1:Dev_weights.rowKeySet())
                    for(int cond2:Dev_weights.row(cond1).keySet())
                    {

                        double update = ccDev.get(cond1,cond2);
                        loss += regLc2*update*update + regLc1*update;

                        double delta_c = eujc* Dev_weights.get(cond1,cond2)- regLc2*update - regLc1;
                        update += lRate*delta_c;
                        ccDev.set(cond1,cond2,update);

                    }


            }



        }
    }

    protected double predict(int u, int j, int c, boolean exclude, int excluded_item) throws Exception {


        Collection<Integer> nns = knn > 0 ? itemNNs.get(j) : allItems;
        SparseVector Ru = userCache.get(u);

        double pred = 0;
        for (int k : nns) {
            if (Ru.contains(k)) {
                if(exclude==true && k == excluded_item)
                    continue;
                else {
                    // extract a random contextual rating by user u and item k
                    String key=u+","+k;
                    int uiid=rateDao.getUserItemId(key);
                    List<Integer> ctxid=this.trainMatrix.getColumns(uiid);

                    Random r = new Random();
                    int index = r.nextInt(ctxid.size());
                    int ctx=ctxid.get(index);

                    // get rating for u, k, ctx
                    double ruk = this.trainMatrix.get(uiid, ctx);
                    double dev_c = getDeviation(ctx, c);

                    pred += (ruk + dev_c) * W.get(k, j);
                }
            }
        }

        return pred;

    }

    private double getDeviation(int from, int to){
            String[] sfrom=rateDao.getContextId(from).split(",");
            String[] sto=rateDao.getContextId(to).split(",");
            double sum=0;
            for(int i=0;i<sfrom.length;++i){
                sum+=ccDev.get(Integer.valueOf(sfrom[i]), Integer.valueOf(sto[i]));
            }
            return sum;
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

