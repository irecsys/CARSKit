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
public class CSLIM_LCS extends CSLIM {
    private DenseMatrix W;

    // item's nearest neighbors for kNN > 0
    private Multimap<Integer, Integer> itemNNs;

    // item's nearest neighbors for kNN <=0, i.e., all other items
    private List<Integer> allItems;

    public CSLIM_LCS(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isRankingPred = true;
        isCARSRecommender=false; // this option is used to allow the algorithm to call 2D rating matrix "train"
        this.algoName = "CSLIM_LCS";

        regLw1 =  algoOptions.getFloat("-lw1");
        regLw2 =  algoOptions.getFloat("-lw2");

        knn = algoOptions.getInt("-k");
    }

    @Override
    protected void initModel() throws Exception {
        super.initModel();

        cfMatrix_LCS=new DenseMatrix(numConditions, numFactors);
        cfMatrix_LCS.init();

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
                List<Integer> conditions=getConditions(c);
                double simc=1.0;
                for(int i=0;i<conditions.size();++i)
                {
                    int index1=conditions.get(i);
                    int index2=EmptyContextConditions.get(i);

                    double sim=1.0;
                    if(index1!=index2) {
                        sim=DenseMatrix.rowMult(cfMatrix_LCS, index1, cfMatrix_LCS, index2);
                        // control the bounds so that sim will not explode
                        sim=(sim>1.0)?1.0-lowbound:sim;
                        sim=(sim<-1.0)?-1.0+lowbound:sim;
                        toBeUpdated.put(index1,index2,sim);
                        simc*=sim;
                    }
                    loss += regC * sim * sim;
                }

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
                //System.out.println("pred: "+pred+", sim: "+simc);
                pred = pred*simc;
                double eujc = rujc - pred;
                loss += eujc * eujc;

                // update similarity values
                // in LCS, it is to update the vector representations
                if(toBeUpdated.size()>0) {
                    for (int index1 : toBeUpdated.rowKeySet())
                        for (int index2 : toBeUpdated.row(index1).keySet()) {
                            // index1 and index2 are pairwise
                            // for each index2, there is only one index2
                            for (int f = 0; f < numFactors; f++) {
                                double c1f = cfMatrix_LCS.get(index1,f);
                                double c2f = cfMatrix_LCS.get(index2,f);
                                double sim = toBeUpdated.get(index1,index2);
                                double delta_c1 = eujc*rating*simc*c2f/sim - regC*c1f;
                                double delta_c2 = eujc*rating*simc*c1f/sim - regC*c2f;

                                cfMatrix_LCS.add(index1,f,lRate*delta_c1);
                                cfMatrix_LCS.add(index2,f,lRate*delta_c2);

                                loss += regC* c1f * c1f + regC * c2f * c2f;
                            }
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
        double sim=1.0;
        for(int i=0;i<conditions.size();++i)
        {
            double[] dv1=cfMatrix_LCS.row(conditions.get(i)).getData();
            double[] dv2=cfMatrix_LCS.row(EmptyContextConditions.get(i)).getData();
            double sum1=0,sum2=0;
            for(int h=0;h<dv1.length;++h){
                sum1+=dv1[h]*dv1[h];
                sum2+=dv2[h]*dv2[h];
            }
            sum1=Math.sqrt(sum1);
            sum2=Math.sqrt(sum2);
            sim*=DenseMatrix.rowMult(cfMatrix_LCS, conditions.get(i), cfMatrix_LCS, EmptyContextConditions.get(i))/(sum1*sum2);
        }

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

