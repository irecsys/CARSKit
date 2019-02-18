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
public class GCSLIM_ICS extends CSLIM {
    private DenseMatrix W;

    // item's nearest neighbors for kNN > 0
    private Multimap<Integer, Integer> itemNNs;

    // item's nearest neighbors for kNN <=0, i.e., all other items
    private List<Integer> allItems;

    public GCSLIM_ICS(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isRankingPred = true;
        isCARSRecommender=false; // this option is used to allow the algorithm to call 2D rating matrix "train"
        this.algoName = "GCSLIM_ICS";

        regLw1 = algoOptions.getFloat("-lw1");
        regLw2 = algoOptions.getFloat("-lw2");

        knn = algoOptions.getInt("-k");
    }

    @Override
    protected void initModel() throws Exception {
        super.initModel();

        ccMatrix_ICS=new SymmMatrix(numConditions);
        for(int i=0;i<numConditions;++i)
            for(int j=0;j<numConditions;++j)
                ccMatrix_ICS.set(i,j,1.0);

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

                HashBasedTable<Integer, Integer, Double> toBeUpdated_sim_factor = HashBasedTable.create();
                HashBasedTable<Integer, Integer, Double> toBeUpdated_w_factor = HashBasedTable.create();
                List<Integer> conditions=getConditions(c);


                Collection<Integer> nns = knn > 0 ? itemNNs.get(j) : allItems;
                SparseVector Ru = userCache.get(u);
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
                            List<Integer> conditions_from=getConditions(ctx);

                            // get rating for u, k, ctx
                            double ruk = this.trainMatrix.get(uiid, ctx);
                            double acc = ruk * W.get(k, j);


                            // calculate similarity
                            double simc=1.0;
                            for(int i=0;i<conditions.size();++i)
                            {
                                int index1=conditions.get(i);
                                int index2=conditions_from.get(i);

                                double sim=1.0;
                                if(index1!=index2) {
                                    sim = ccMatrix_ICS.get(index1, index2);
                                    double update_factor=acc/sim;
                                    if(toBeUpdated_sim_factor.contains(index1,index2))
                                        update_factor+=toBeUpdated_sim_factor.get(index1,index2);
                                    toBeUpdated_sim_factor.put(index1,index2,update_factor);
                                    simc*=sim;
                                }
                                loss += regC * sim * sim;
                            }

                            for(int row:toBeUpdated_sim_factor.rowKeySet())
                                for(int col:toBeUpdated_sim_factor.row(row).keySet())
                            {
                                    toBeUpdated_sim_factor.put(row,col,simc*toBeUpdated_sim_factor.get(row,col));

                            }

                            toBeUpdated_w_factor.put(k,j,ruk*simc);
                            pred+=acc*simc;

                        }
                    }
                }
                double rating=pred;
                double eujc = rujc - pred;
                loss += eujc * eujc;


                // update similarity values
                if(toBeUpdated_sim_factor.size()>0){
                    for (int index1 : toBeUpdated_sim_factor.rowKeySet())
                        for (int index2 : toBeUpdated_sim_factor.row(index1).keySet()) {
                            double update = this.ccMatrix_ICS.get(index1, index2);
                            update += lRate * (eujc * toBeUpdated_sim_factor.get(index1, index2) - regC * update);
                            ccMatrix_ICS.set(index1, index2, update);
                        }
                }

                // start updating W
                if(toBeUpdated_w_factor.size()>0){
                    for (int index1 : toBeUpdated_w_factor.rowKeySet())
                        for (int index2 : toBeUpdated_w_factor.row(index1).keySet()) {
                            double update = W.get(index1,index2);
                            loss += regLw2*update*update + regLw1*update;
                            double delta_w = eujc*toBeUpdated_w_factor.get(index1,index2) - regLw2*update - regLw1;
                            update += lRate*delta_w;
                            W.set(index1,index2,update);
                        }
                }

            }
        }
    }

    protected double predict(int u, int j, int c, boolean exclude, int excluded_item) throws Exception {


        Collection<Integer> nns = knn > 0 ? itemNNs.get(j) : allItems;
        SparseVector Ru = userCache.get(u);

        List<Integer> conditions=getConditions(c);

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
                    List<Integer> conditions_from=getConditions(ctx);

                    // get rating for u, k, ctx
                    double ruk = this.trainMatrix.get(uiid, ctx);

                    double sim=1.0;
                    for(int i=0;i<conditions.size();++i)
                    {
                        sim*=ccMatrix_ICS.get(conditions.get(i), conditions_from.get(i));
                    }

                    pred += ruk * W.get(k, j)*sim;
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

