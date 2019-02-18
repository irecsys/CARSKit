package carskit.alg.cars.adaptation.dependent.sim;

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
import librec.data.*;

import java.util.*;
import java.util.Map.Entry;

/**
 * Yong Zheng, Bamshad Mobasher, Robin Burke. "Integrating Context Similarity with Sparse Linear Recommendation Model", Proceedings of the 23rd Conference on User Modeling, Adaptation and Personalization (UMAP), pp. 370-376, Dublin, Ireland, June 2015
 */


@Configuration("binThold, knn, regLw2, regLw1, similarity, iters, rc")
public class CSLIM_MCS extends CSLIM {
    private DenseMatrix W;

    // item's nearest neighbors for kNN > 0
    private Multimap<Integer, Integer> itemNNs;

    // item's nearest neighbors for kNN <=0, i.e., all other items
    private List<Integer> allItems;

    private double upbound;
    private double lowbound;

    public CSLIM_MCS(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isRankingPred = true;
        isCARSRecommender=false; // this option is used to allow the algorithm to call 2D rating matrix "train"
        this.algoName = "CSLIM_MCS";

        regLw1 = algoOptions.getFloat("-lw1");
        regLw2 = algoOptions.getFloat("-lw2");

        knn = algoOptions.getInt("-k");
    }

    @Override
    protected void initModel() throws Exception {
        super.initModel();

        upbound = 1.0/Math.sqrt(rateDao.numContextDims());
        lowbound = 1.0/Math.pow(10, 100);

        cVector_MCS=new librec.data.DenseVector(numConditions);
        cVector_MCS.init(upbound);

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

                HashBasedTable<Integer, Integer, Double> toBeUpdated = HashBasedTable.create();
                double simc=1.0;
                List<Integer> conditions=getConditions(c);
                double dist=0;
                for(int i=0;i<conditions.size();++i) {
                    int index1=conditions.get(i);
                    int index2=EmptyContextConditions.get(i);
                    double pos1=cVector_MCS.get(index1);
                    double pos2=cVector_MCS.get(index2);
                    double diff = pos1-pos2;
                    dist += Math.pow(diff, 2);
                    if(index1!=index2)
                        toBeUpdated.put(index1,index2,diff);

                    loss += regC*pos1*pos1 + regC*pos2*pos2;
                }

                dist = Math.sqrt(dist);
                simc = 1-dist;

                Collection<Integer> nns = knn > 0 ? itemNNs.get(j) : allItems;
                SparseVector Ru = userCache.get(u);
                double pred = 0;
                for (int k : nns) {
                    if (Ru.contains(k)) {
                        if(k != j) {
                            double ruk = Ru.get(k);
                            pred += ruk * W.get(k, j);
                        }
                    }
                }
                double rating=pred;
                pred = pred*simc;
                double eujc = rujc - pred;
                loss += eujc * eujc;


                // update similarity values
                if(toBeUpdated.size()>0) {
                    for (int index1 : toBeUpdated.rowKeySet())
                        for (int index2 : toBeUpdated.row(index1).keySet()) {
                            double pos1 = cVector_MCS.get(index1);
                            double pos2 = cVector_MCS.get(index2);

                            if(dist==0)
                                dist=lowbound;

                            double pos1_update = pos1 + lRate*(eujc*rating*toBeUpdated.get(index1,index2)/dist - regC*pos1);
                            double pos2_update = pos2 - lRate*(eujc*rating*toBeUpdated.get(index1,index2)/dist + regC*pos2);

                            // In this paper, we use a simple rule as constraint to limit the distance value.

                            pos1_update = (pos1_update<0)?lowbound:pos1_update;
                            pos1_update = (pos1_update>upbound)?upbound-lowbound:pos1_update;

                            pos2_update = (pos2_update<0)?lowbound:pos2_update;
                            pos2_update = (pos2_update>upbound)?upbound-lowbound:pos2_update;

                            cVector_MCS.set(index1,pos1_update);
                            cVector_MCS.set(index2,pos2_update);
                        }
                }

                // start updating W
                double sum_w=0;
                for (int k : nns) {
                    double update=W.get(k, j);
                    sum_w += update;

                    loss += regLw2*update*update + regLw1*update;

                    double delta_w = eujc*Ru.get(k)*simc - regLw2*update - regLw1;
                    update += lRate*delta_w;
                    W.set(k,j,update);
                }


            }



        }
    }

    protected double predict(int u, int j, int c, boolean exclude, int excluded_item) throws Exception {


        Collection<Integer> nns = knn > 0 ? itemNNs.get(j) : allItems;
        SparseVector Ru = userCache.get(u);


        List<Integer> conditions=getConditions(c);
        double dist=0;
        for(int i=0;i<conditions.size();++i) {
            int index1=conditions.get(i);
            int index2=EmptyContextConditions.get(i);
            dist += Math.pow(cVector_MCS.get(index1) - cVector_MCS.get(index2), 2);
        }
        dist = Math.sqrt(dist);
        double sim=1-dist;

        double pred = 0;
        for (int k : nns) {
            if (Ru.contains(k)) {
                if(exclude==true && k == excluded_item)
                    continue;
                else {
                    double ruk = Ru.get(k);
                    pred += ruk * W.get(k, j);
                }
            }
        }
        pred = pred*sim;
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

