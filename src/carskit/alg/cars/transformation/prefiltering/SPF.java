
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

import carskit.data.structure.SparseMatrix;
import carskit.generic.ContextRecommender;

import java.util.*;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;
import happy.coding.io.FileIO;
import happy.coding.io.Lists;
import happy.coding.io.Logs;
import happy.coding.io.Strings;
import carskit.eval.Measures;
import happy.coding.math.Stats;
import librec.data.*;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table.Cell;
import com.google.common.collect.HashMultimap;
import librec.data.SparseVector;
import librec.data.DenseVector;
import librec.data.DenseMatrix;

/**
 * Victor Codina, Francesco Ricci, Luigi Ceccaroni. Distributional semantic pre-filtering in context-aware recommender systems. User Model. User-Adapt. Interact. 26(1): 1-32 (2016)
 * <p></p>
 * Note: The setting for this algorithm: 1). you can choose either item-based or user-based semantic similarity; 2). global similarity threshold; 3). direct context similarity
 *
 * @author Yong Zheng
 *
 */
public class SPF extends ContextRecommender {

    protected double th;
    protected int itembased;
    protected double mean, beta;
    protected HashMap<Integer, Double> bu;
    protected HashMap<Integer, Double> bi;
    //protected SymmMatrix SimMatrix;
    protected DenseMatrix C, E;
    protected int f, t;
    protected double r, l;

    public SPF(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isCARSRecommender=false; // this option is used to allow the algorithm to call 2D rating matrix "train"
        this.algoName = "SPF";

        th = algoOptions.getDouble("-th");
        itembased = algoOptions.getInt("-i");
        beta = algoOptions.getDouble("-b");

        f=algoOptions.getInt("-f",10);
        t=algoOptions.getInt("-t",90);
        r=algoOptions.getDouble("-r",0.01);
        l=algoOptions.getDouble("-l",0.01);

        // calculate globalMean, user and item bias
        mean=this.globalMean;
        bu=new HashMap<Integer, Double>();
        bi=new HashMap<Integer, Double>();
        if(train==null)
            train=rateDao.toTraditionalSparseMatrix(trainMatrix);
        for(int i=0;i<numUsers;++i){
            SparseVector sv=train.row(i);
            double avgu = (sv.getCount()>0) ? sv.mean() : this.globalMean;
            bu.put(i,avgu-mean);
        }
        for(int i=0;i<numItems;++i){
            SparseVector sv=train.column(i);
            double avgi = (sv.getCount()>0) ? sv.mean() : this.globalMean;
            bi.put(i,avgi-mean);
        }


        // initialization: context similarity
        //SimMatrix = new SymmMatrix(numConditions);

        // calculate and store context similarity
        SparseMatrix sm;
        DenseVector cBias, eBias;
        cBias= new DenseVector(numConditions);
        if(itembased == 1) {
            sm = getCIMatrix();
            eBias = new DenseVector(numItems);
        }
        else {
            sm = getCUMatrix();
            eBias = new DenseVector(numUsers);
        }
        cBias.init(initMean, initStd);
        eBias.init(initMean, initStd);


        C = new DenseMatrix(sm.numRows(), f);
        E = new DenseMatrix(sm.numColumns(), f);

        // initialize model
        if (initByNorm) {
            C.init(initMean, initStd);
            E.init(initMean, initStd);
        } else {
            C.init(); // P.init(smallValue);
            E.init(); // Q.init(smallValue);
        }

        trainMF(sm, cBias, eBias, C, E, f, t, l, r, r, r);
        // after training, the C matrix is the context matrix.

        /*
        // put each pair of similarity into SimMatrix
        for(int i=0;i<numConditions;++i)
            for(int j=i+1; j<numConditions;++j)
                SimMatrix.set(i, j,librec.data.DenseMatrix.rowMult(P, i, P, j));
        */
    }


    protected void trainMF(SparseMatrix train,DenseVector cBias, DenseVector eBias, DenseMatrix C, DenseMatrix E,
                           int numFactors, int numIters, double lRate, double regB, double regU, double regI){

        for (int iter = 1; iter <= numIters; iter++) {
            loss = 0;
            for (MatrixEntry me : train) {

                int c = me.row(); // user
                int j = me.column(); // item
                double rcj = me.get();

                double pred = globalMean + cBias.get(c) + eBias.get(j) + librec.data.DenseMatrix.rowMult(C, c, E, j);
                double ecj = rcj - pred;

                loss += ecj * ecj;

                // update factors
                double bc = cBias.get(c);
                double sgd = ecj - regB * bc;
                cBias.add(c, lRate * sgd);

                loss += regB * bc * bc;

                double bj = eBias.get(j);
                sgd = ecj - regB * bj;
                eBias.add(j, lRate * sgd);

                loss += regB * bj * bj;

                for (int f = 0; f < numFactors; f++) {
                    double pcf = C.get(c, f);
                    double qjf = E.get(j, f);

                    double delta_c = ecj * qjf - regU * pcf;
                    double delta_j = ecj * pcf - regI * qjf;

                    C.add(c, f, lRate * delta_c);
                    E.add(j, f, lRate * delta_j);

                    loss += regU * pcf * pcf + regI * qjf * qjf;
                }

            }
            loss *= 0.5;

        }// end of training
    }

    protected SparseMatrix getCUMatrix()
    {
        // Table {row-id, col-id, rate}
        Table<Integer, Integer, Double> dataTable_cu = HashBasedTable.create();
        Table<Integer, Integer, Double> dataTable_cu_count = HashBasedTable.create();

        // Map {col-id, multiple row-id}: used to fast build a rating matrix
        Multimap<Integer, Integer> colMap = HashMultimap.create();

        // read data to have a list of rating profiles for each uc pair
        for (MatrixEntry me : trainMatrix){
            int ui = me.row(); // user-item
            int u= rateDao.getUserIdFromUI(ui);
            int j= rateDao.getItemIdFromUI(ui);

            int ctx = me.column(); // context

            Collection<Integer> cs= rateDao.getContextConditionsList().get(ctx);

            double rujc = me.get();
            double bui = mean + bu.get(u) + bi.get(j);

            for(int c:cs) {
                if (dataTable_cu.contains(c, u)) {
                    dataTable_cu.put(c, u, dataTable_cu.get(c, u) + rujc - bui);
                    dataTable_cu_count.put(c, u, dataTable_cu_count.get(c, u) + 1.0);
                } else {
                    dataTable_cu.put(c, u, rujc - bui);
                    dataTable_cu_count.put(c, u, 1.0);
                }
            }
        }

        // formulate sparse matrix in order to perform SVD
        for(Cell<Integer, Integer, Double> cell: dataTable_cu.cellSet()){
            int c = cell.getRowKey();
            int u = cell.getColumnKey();
            dataTable_cu.put(c, u, cell.getValue()/(beta+dataTable_cu_count.get(c, u)));
            colMap.put(u, c);
        }
        //Logs.info("numConditions = " + numConditions+", datatable.row = "+dataTable_cu.rowKeySet().size());
        return new SparseMatrix(numConditions, numUsers, dataTable_cu, colMap);
    }

    protected SparseMatrix getCIMatrix()
    {
        // Table {row-id, col-id, rate}
        Table<Integer, Integer, Double> dataTable_ci = HashBasedTable.create();
        Table<Integer, Integer, Double> dataTable_ci_count = HashBasedTable.create();

        // Map {col-id, multiple row-id}: used to fast build a rating matrix
        Multimap<Integer, Integer> colMap = HashMultimap.create();

        // read data to have a list of rating profiles for each uc pair

        for (MatrixEntry me : trainMatrix){
            int ui = me.row(); // user-item
            int u= rateDao.getUserIdFromUI(ui);
            int j= rateDao.getItemIdFromUI(ui);
            int ctx = me.column(); // context

            Collection<Integer> cs= rateDao.getContextConditionsList().get(ctx);

            double rujc = me.get();
            double bui = mean + bu.get(u) + bi.get(j);

            for(int c:cs) {
                if (dataTable_ci.contains(c, j)) {
                    dataTable_ci.put(c, j, dataTable_ci.get(c, j) + rujc - bui);
                    dataTable_ci_count.put(c, j, dataTable_ci_count.get(c, j) + 1.0);
                } else {
                    dataTable_ci.put(c, j, rujc - bui);
                    dataTable_ci_count.put(c, j, 1.0);
                }
            }
        }

        // formulate sparse matrix in order to perform SVD
        for(Cell<Integer, Integer, Double> cell: dataTable_ci.cellSet()){
            int c = cell.getRowKey();
            int j = cell.getColumnKey();
            dataTable_ci.put(c, j, cell.getValue()/(beta+dataTable_ci_count.get(c, j)));
            colMap.put(j, c);
        }
        return new SparseMatrix(numConditions, numItems, dataTable_ci,colMap);

    }

    protected DenseVector getContextVector(int ctx){
        Collection<Integer> css= rateDao.getContextConditionsList().get(ctx);
        DenseVector v_ctx = new DenseVector(C.numColumns());
        for(int index_cs:css){
            v_ctx=v_ctx.add(C.row(index_cs));
        }
        for(int i=0;i<C.numColumns();++i)
            v_ctx.set(i,v_ctx.get(i)/css.size());
        return v_ctx;
    }

    protected double cosineSimilarity(DenseVector v1, DenseVector v2){
        int size = v1.getData().length;
        double sum1=0, sum2=0, sum3=0;
        for(int i=0;i<size;++i)
        {
            sum1+=v1.get(i)*v2.get(i);
            sum2+=v1.get(i)*v1.get(i);
            sum3+=v2.get(i)*v2.get(i);
        }

        return sum1/(Math.sqrt(sum2)*Math.sqrt(sum3));
    }

    protected SparseMatrix getUIMatrix(int ctx)
    {
        DenseVector vc_target = getContextVector(ctx);

        // Table {row-id, col-id, rate}
        Table<Integer, Integer, Double> dataTable_ui = HashBasedTable.create();
        Table<Integer, Integer, Double> dataTable_ui_count = HashBasedTable.create();

        // Map {col-id, multiple row-id}: used to fast build a rating matrix
        Multimap<Integer, Integer> colMap = HashMultimap.create();

        // read data to have a list of rating profiles for each uc pair
        for (MatrixEntry me : trainMatrix) {
            int ui = me.row(); // user-item
            int u = rateDao.getUserIdFromUI(ui);
            int j = rateDao.getItemIdFromUI(ui);
            int c = me.column(); // context
            DenseVector vc_current = getContextVector(c);
            double sim = cosineSimilarity(vc_target, vc_current);
            if (sim >= th) {
                double rujc = me.get();
                if (dataTable_ui.contains(u, j)) {
                    dataTable_ui.put(u, j, dataTable_ui.get(u, j));
                    dataTable_ui_count.put(u, j, dataTable_ui_count.get(u, j) + 1.0);
                } else {
                    dataTable_ui.put(u, j, rujc);
                    dataTable_ui_count.put(u, j, 1.0);
                }
            }

            // formulate sparse matrix in order to perform SVD
            for (Cell<Integer, Integer, Double> cell : dataTable_ui.cellSet()) {
                int uu = cell.getRowKey();
                int jj = cell.getColumnKey();
                dataTable_ui.put(uu, jj, cell.getValue() / dataTable_ui_count.get(uu, jj));
                colMap.put(jj, uu);
            }
        }

        return new SparseMatrix(numUsers, numItems, dataTable_ui, colMap);
    }

    @Override
    protected Map<Measure, Double> evalRankings() throws Exception {

        HashMap<Integer, HashMultimap<Integer, Integer>> cuiList=rateDao.getCtxUserList(testMatrix, binThold);
        HashMap<Integer, HashMultimap<Integer, Integer>> cuiList_train=rateDao.getCtxUserList(trainMatrix);
        int capacity = cuiList.keySet().size();

        // initialization capacity to speed up
        List<Double> ds5 = new ArrayList<>(isDiverseUsed ? capacity : 0);
        List<Double> ds10 = new ArrayList<>(isDiverseUsed ? capacity : 0);
        List<Double> dsN = new ArrayList<>(isDiverseUsed ? capacity : 0);

        List<Double> precs5 = new ArrayList<>(capacity);
        List<Double> precs10 = new ArrayList<>(capacity);
        List<Double> precsN = new ArrayList<>(capacity);
        List<Double> recalls5 = new ArrayList<>(capacity);
        List<Double> recalls10 = new ArrayList<>(capacity);
        List<Double> recallsN = new ArrayList<>(capacity);
        List<Double> aps5 = new ArrayList<>(capacity);
        List<Double> aps10 = new ArrayList<>(capacity);
        List<Double> apsN = new ArrayList<>(capacity);
        List<Double> rrs5 = new ArrayList<>(capacity);
        List<Double> rrs10 = new ArrayList<>(capacity);
        List<Double> rrsN = new ArrayList<>(capacity);
        List<Double> aucs5 = new ArrayList<>(capacity);
        List<Double> aucs10 = new ArrayList<>(capacity);
        List<Double> aucsN = new ArrayList<>(capacity);
        List<Double> ndcgs5 = new ArrayList<>(capacity);
        List<Double> ndcgs10 = new ArrayList<>(capacity);
        List<Double> ndcgsN = new ArrayList<>(capacity);

        // candidate items for all users: here only training items
        // use HashSet instead of ArrayList to speedup removeAll() and contains() operations: HashSet: O(1); ArrayList: O(log n).
        Set<Integer> candItems = rateDao.getItemList(trainMatrix);

        List<String> preds = null;
        String toFile = null;
        int numTopNRanks = numRecs < 0 ? 10 : numRecs;
        if (isResultsOut) {
            preds = new ArrayList<String>(1500);
            preds.add("# userId: recommendations in (itemId, ranking score) pairs, where a correct recommendation is denoted by symbol *."); // optional: file header
            toFile = workingPath
                    + String.format("%s-top-%d-items%s.txt", algoName, numTopNRanks, foldInfo); // the output-file name
            FileIO.deleteFile(toFile); // delete possibly old files
        }

        if (verbose)
            Logs.debug("{}{} has candidate items: {}", algoName, foldInfo, candItems.size());

        // ignore items for all users: most popular items
        if (numIgnore > 0) {
            List<Map.Entry<Integer, Integer>> itemDegs = new ArrayList<>();
            for (Integer j : candItems) {
                itemDegs.add(new AbstractMap.SimpleImmutableEntry<Integer, Integer>(j, rateDao.getRatingCountByItem(trainMatrix,j)));
            }
            Lists.sortList(itemDegs, true);
            int k = 0;
            for (Map.Entry<Integer, Integer> deg : itemDegs) {

                // ignore these items from candidate items
                candItems.remove(deg.getKey());
                if (++k >= numIgnore)
                    break;
            }
        }

        // for each context
        for (int ctx:cuiList.keySet()) {

            Multimap<Integer, Integer> uis = cuiList.get(ctx);

            int u_capacity = uis.keySet().size();

            List<Double> c_ds5 = new ArrayList<>(isDiverseUsed ? u_capacity : 0);
            List<Double> c_ds10 = new ArrayList<>(isDiverseUsed ? u_capacity : 0);
            List<Double> c_dsN = new ArrayList<>(isDiverseUsed ? u_capacity : 0);

            List<Double> c_precs5 = new ArrayList<>(u_capacity);
            List<Double> c_precs10 = new ArrayList<>(u_capacity);
            List<Double> c_precsN = new ArrayList<>(u_capacity);
            List<Double> c_recalls5 = new ArrayList<>(u_capacity);
            List<Double> c_recalls10 = new ArrayList<>(u_capacity);
            List<Double> c_recallsN = new ArrayList<>(u_capacity);
            List<Double> c_aps5 = new ArrayList<>(u_capacity);
            List<Double> c_aps10 = new ArrayList<>(u_capacity);
            List<Double> c_apsN = new ArrayList<>(u_capacity);
            List<Double> c_rrs5 = new ArrayList<>(u_capacity);
            List<Double> c_rrs10 = new ArrayList<>(u_capacity);
            List<Double> c_rrsN = new ArrayList<>(u_capacity);
            List<Double> c_aucs5 = new ArrayList<>(u_capacity);
            List<Double> c_aucs10 = new ArrayList<>(u_capacity);
            List<Double> c_aucsN = new ArrayList<>(u_capacity);
            List<Double> c_ndcgs5 = new ArrayList<>(u_capacity);
            List<Double> c_ndcgs10 = new ArrayList<>(u_capacity);
            List<Double> c_ndcgsN = new ArrayList<>(u_capacity);
            HashMultimap<Integer, Integer> uList_train = (cuiList_train.containsKey(ctx))?cuiList_train.get(ctx):HashMultimap.<Integer, Integer>create();

            // for each ctx, we build a 2D rating matrix -- only users and items
            SparseMatrix UIM = getUIMatrix(ctx);
            userBias= new DenseVector(numUsers);
            itemBias= new DenseVector(numItems);
            userBias.init(initMean, initStd);
            itemBias.init(initMean, initStd);
            trainMF(UIM, userBias, itemBias, P, Q, numFactors, numIters, lRate, regB, regU, regI);

            // for each user
            for (int u : uis.keySet()) {

                if (verbose && ((u + 1) % 100 == 0))
                    Logs.debug("{}{} evaluates progress: {} / {}", algoName, foldInfo, u + 1, capacity);

                // number of candidate items for all users
                int numCands = candItems.size();

                // get positive items from test matrix
                Collection<Integer> posItems = uis.get(u);
                List<Integer> correctItems = new ArrayList<>();

                // intersect with the candidate items
                for (Integer j : posItems) {
                    if (candItems.contains(j))
                        correctItems.add(j);
                }

                if (correctItems.size() == 0)
                    continue; // no testing data for user u

                // remove rated items from candidate items
                Set<Integer> ratedItems = (uList_train.containsKey(u))?uList_train.get(u):new HashSet<Integer>();

                // predict the ranking scores (unordered) of all candidate items
                List<Map.Entry<Integer, Double>> itemScores = new ArrayList<>(Lists.initSize(candItems));
                for (final Integer j : candItems) {
                    if (!ratedItems.contains(j)) {
                        final double rank = ranking(u, j, ctx);
                        if (!Double.isNaN(rank)) {
                            if(rank>binThold)
                                itemScores.add(new AbstractMap.SimpleImmutableEntry<Integer, Double>(j, rank));
                        }
                    } else {
                        numCands--;
                    }
                }

                if (itemScores.size() == 0)
                    continue; // no recommendations available for user u

                // order the ranking scores from highest to lowest: List to preserve orders
                Lists.sortList(itemScores, true);
                List<Map.Entry<Integer, Double>> recomd = (numRecs <= 0 || itemScores.size() <= numRecs) ? itemScores
                        : itemScores.subList(0, numRecs);

                List<Integer> rankedItems = new ArrayList<>();
                StringBuilder sb = new StringBuilder();
                int count = 0;
                for (Map.Entry<Integer, Double> kv : recomd) {
                    Integer item = kv.getKey();
                    rankedItems.add(item);

                    if (isResultsOut && count < numTopNRanks) {
                        // restore back to the original item id
                        sb.append("(").append(rateDao.getItemId(item));

                        if (posItems.contains(item))
                            sb.append("*"); // indicating correct recommendation

                        sb.append(", ").append(kv.getValue().floatValue()).append(")");

                        if (++count >= numTopNRanks)
                            break;
                        if (count < numTopNRanks)
                            sb.append(", ");
                    }
                }

                int numDropped = numCands - rankedItems.size();
                double AUC = Measures.AUC(rankedItems, correctItems, numDropped);
                double AP = Measures.AP(rankedItems, correctItems);
                double nDCG = Measures.nDCG(rankedItems, correctItems);
                double RR = Measures.RR(rankedItems, correctItems);

                List<Integer> cutoffs = Arrays.asList(5, 10, numRecs);
                Map<Integer, Double> precs = Measures.PrecAt(rankedItems, correctItems, cutoffs);
                Map<Integer, Double> recalls = Measures.RecallAt(rankedItems, correctItems, cutoffs);
                Map<Integer, Double> aucs = Measures.AUCAt(rankedItems, correctItems, numDropped,cutoffs);
                Map<Integer, Double> aps = Measures.APAt(rankedItems, correctItems, cutoffs);
                Map<Integer, Double> ndcgs = Measures.nDCGAt(rankedItems, correctItems, cutoffs);
                Map<Integer, Double> rrs = Measures.RRAt(rankedItems, correctItems, cutoffs);

                c_precs5.add(precs.get(5));
                c_precs10.add(precs.get(10));
                c_precsN.add(precs.get(numRecs));
                c_recalls5.add(recalls.get(5));
                c_recalls10.add(recalls.get(10));
                c_recallsN.add(recalls.get(numRecs));


                c_aucs5.add(aucs.get(5));
                c_aps5.add(aps.get(5));
                c_rrs5.add(rrs.get(5));
                c_ndcgs5.add(ndcgs.get(5));
                c_aucs10.add(aucs.get(10));
                c_aps10.add(aps.get(10));
                c_rrs10.add(rrs.get(10));
                c_ndcgs10.add(ndcgs.get(10));
                c_aucsN.add(aucs.get(numRecs));
                c_apsN.add(aps.get(numRecs));
                c_rrsN.add(rrs.get(numRecs));
                c_ndcgsN.add(ndcgs.get(numRecs));


                // diversity
                if (isDiverseUsed) {
                    double d5 = diverseAt(rankedItems, 5);
                    double d10 = diverseAt(rankedItems, 10);
                    double dN = diverseAt(rankedItems, numRecs);

                    c_ds5.add(d5);
                    c_ds10.add(d10);
                    c_dsN.add(dN);
                }

                // output predictions
                if (isResultsOut) {
                    // restore back to the original user id
                    preds.add(rateDao.getUserId(u) + ", " + rateDao.getContextSituationFromInnerId(ctx) + ": " + sb.toString());
                    if (preds.size() >= 1000) {
                        FileIO.writeList(toFile, preds, true);
                        preds.clear();
                    }
                }
            } // end a context

            // calculate metrics for a specific user averaged by contexts
            ds5.add(isDiverseUsed ? Stats.mean(c_ds5) : 0.0);
            ds10.add(isDiverseUsed ? Stats.mean(c_ds10) : 0.0);
            dsN.add(isDiverseUsed ? Stats.mean(c_dsN) : 0.0);
            precs5.add(Stats.mean(c_precs5));
            precs10.add(Stats.mean(c_precs10));
            precsN.add(Stats.mean(c_precsN));
            recalls5.add(Stats.mean(c_recalls5));
            recalls10.add(Stats.mean(c_recalls10));
            recallsN.add(Stats.mean(c_recallsN));
            aucs5.add(Stats.mean(c_aucs5));
            ndcgs5.add(Stats.mean(c_ndcgs5));
            aps5.add(Stats.mean(c_aps5));
            rrs5.add(Stats.mean(c_rrs5));
            aucs10.add(Stats.mean(c_aucs10));
            ndcgs10.add(Stats.mean(c_ndcgs10));
            aps10.add(Stats.mean(c_aps10));
            rrs10.add(Stats.mean(c_rrs10));
            aucsN.add(Stats.mean(c_aucsN));
            ndcgsN.add(Stats.mean(c_ndcgsN));
            apsN.add(Stats.mean(c_apsN));
            rrsN.add(Stats.mean(c_rrsN));
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



        // write results out first
        if (isResultsOut && preds.size() > 0) {
            FileIO.writeList(toFile, preds, true);
            Logs.debug("{}{} has writeen item recommendations to {}", algoName, foldInfo, toFile);
        }

        // measure the performance
        Map<Measure, Double> measures = new HashMap<>();
        measures.put(Measure.D5, isDiverseUsed ? Stats.mean(ds5) : 0.0);
        measures.put(Measure.D10, isDiverseUsed ? Stats.mean(ds10) : 0.0);
        measures.put(Measure.DN, isDiverseUsed ? Stats.mean(dsN) : 0.0);
        measures.put(Measure.Pre5, Stats.mean(precs5));
        measures.put(Measure.Pre10, Stats.mean(precs10));
        measures.put(Measure.PreN, Stats.mean(precsN));
        measures.put(Measure.Rec5, Stats.mean(recalls5));
        measures.put(Measure.Rec10, Stats.mean(recalls10));
        measures.put(Measure.RecN, Stats.mean(recallsN));
        measures.put(Measure.AUC5, Stats.mean(aucs5));
        measures.put(Measure.NDCG5, Stats.mean(ndcgs5));
        measures.put(Measure.MAP5, Stats.mean(aps5));
        measures.put(Measure.MRR5, Stats.mean(rrs5));
        measures.put(Measure.AUC10, Stats.mean(aucs10));
        measures.put(Measure.NDCG10, Stats.mean(ndcgs10));
        measures.put(Measure.MAP10, Stats.mean(aps10));
        measures.put(Measure.MRR10, Stats.mean(rrs10));
        measures.put(Measure.AUCN, Stats.mean(aucsN));
        measures.put(Measure.NDCGN, Stats.mean(ndcgsN));
        measures.put(Measure.MAPN, Stats.mean(apsN));
        measures.put(Measure.MRRN, Stats.mean(rrsN));

        return measures;
    }

    @Override
    protected Map<Measure, Double> evalRatings() throws Exception {

        List<String> preds = null;
        String toFile = null;
        if (isResultsOut) {
            preds = new ArrayList<String>(1500);
            preds.add("userId\titemId\tcontexts\trating\tprediction"); // optional: file header
            toFile = workingPath + algoName + "-rating-predictions" + foldInfo + ".txt"; // the output-file name
            FileIO.deleteFile(toFile); // delete possibly old files
        }

        double sum_maes = 0, sum_mses = 0, sum_r_maes = 0, sum_r_rmses = 0, sum_perps = 0;
        int numCount = 0, numPEs = 0;

        for(int col: testMatrix.columns()){
            int ctx = col;
            // for each ctx, we build a 2D rating matrix -- only users and items
            SparseMatrix UIM = getUIMatrix(ctx);
            userBias= new DenseVector(numUsers);
            itemBias= new DenseVector(numItems);
            userBias.init(initMean, initStd);
            itemBias.init(initMean, initStd);
            trainMF(UIM, userBias, itemBias, P, Q, numFactors, numIters, lRate, regB, regU, regI);

            for(VectorEntry en:testMatrix.column(ctx)){
                int ui = en.index();
                double rate = en.get();
                int u=rateDao.getUserIdFromUI(ui);
                int j=rateDao.getItemIdFromUI(ui);
                double pred = predict(u,j, ctx, true);
                if (Double.isNaN(pred))
                    continue;

                // perplexity: for some graphic model
                double perp = perplexity(u, j, pred);
                sum_perps += perp;

                // rounding prediction to the closest rating level
                double rPred = Math.round(pred / minRate) * minRate;

                double err = Math.abs(rate - pred); // absolute predictive error
                double r_err = Math.abs(rate - rPred);

                sum_maes += err;
                sum_mses += err * err;

                sum_r_maes += r_err;
                sum_r_rmses += r_err * r_err;

                numCount++;

                // output predictions
                if (isResultsOut) {
                    // restore back to the original user/item id
                    preds.add(rateDao.getUserId(u) + "\t" + rateDao.getItemId(j) + "\t" + rateDao.getContextSituationFromInnerId(ctx) + "\t" + rate + "\t" + (float) pred);
                    if (preds.size() >= 1000) {
                        FileIO.writeList(toFile, preds, true);
                        preds.clear();
                    }
                }
            }
        }

        if (isResultsOut && preds.size() > 0) {
            FileIO.writeList(toFile, preds, true);
            Logs.debug("{}{} has writeen rating predictions to {}", algoName, foldInfo, toFile);
        }

        double mae = sum_maes / numCount;
        double rmse = Math.sqrt(sum_mses / numCount);

        double r_mae = sum_r_maes / numCount;
        double r_rmse = Math.sqrt(sum_r_rmses / numCount);

        Map<Measure, Double> measures = new HashMap<>();
        measures.put(Measure.MAE, mae);
        // normalized MAE: useful for direct comparison among different data sets with distinct rating scales
        measures.put(Measure.NMAE, mae / (maxRate - minRate));
        measures.put(Measure.RMSE, rmse);

        // MAE and RMSE after rounding predictions to the closest rating levels
        measures.put(Measure.rMAE, r_mae);
        measures.put(Measure.rRMSE, r_rmse);

        // measure zero-one loss
        measures.put(Measure.MPE, (numPEs + 0.0) / numCount);

        // perplexity
        if (sum_perps > 0) {
            measures.put(Measure.Perplexity, Math.exp(sum_perps / numCount));
        }

        return measures;
    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        return globalMean + userBias.get(u) + itemBias.get(j) + DenseMatrix.rowMult(P, u, Q, j);
    }

    @Override
    public String toString() {
        return Strings.toString(new Object[]{"numFactors: " + numFactors, "numIter: " + numIters, "lrate: " + initLRate, "maxlrate: " + maxLRate, "regB: " + regB, "regU: " + regU, "regI: " + regI, "regC: " + regC,
                "knn: "+knn, "-i: "+this.itembased, "-b: "+this.beta, "-th: "+this.th, "isBoldDriver: " + isBoldDriver});
    }
}
