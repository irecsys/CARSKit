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

import carskit.data.structure.*;
import carskit.data.structure.SparseMatrix;
import carskit.generic.IterativeRecommender;
import carskit.generic.Recommender;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;
import happy.coding.io.FileIO;
import happy.coding.io.Lists;
import happy.coding.io.Logs;
import happy.coding.io.Strings;
import carskit.eval.Measures;
import happy.coding.math.Stats;
import librec.data.*;
import librec.data.DenseVector;
import librec.data.SparseVector;

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
    private carskit.data.structure.SparseMatrix sm;


    public ExactFiltering(carskit.data.structure.SparseMatrix trainMatrix, carskit.data.structure.SparseMatrix testMatrix, int fold) {

        super(trainMatrix, testMatrix, fold);
        this.algoName = "ExactFiltering";
    }


    @Override
    protected void initModel() throws Exception {
        super.initModel();
        if(isRankingPred==false){
            userCorrs = buildCorrs(true);
            userMeans = new DenseVector(numUsers);
            for (int u = 0; u < numUsers; u++) {
                SparseVector uv = train.row(u);
                userMeans.set(u, uv.getCount() > 0 ? uv.mean() : globalMean);
            }
        }
    }

    @Override
    protected double predict(int a, int t, int c) throws Exception {
        if(isRankingPred) {
            double pred = 0;

            HashMap<Integer, Double> nns = new HashMap<>();
            HashMap<Integer, Double> nns_ratings = new HashMap<>();

            SparseVector sv = userCorrs.row(a);
            for (int u : sv.getIndex()) {
                if (nns.size() >= knn)
                    break;
                else {
                    double sim = sv.get(u);
                    if (sim > 0) {
                        double rate = this.sm.get(u, t);
                        if (rate > 0) {
                            nns.put(u, sim);
                            nns_ratings.put(u, rate);
                        }
                    }
                }
            }


            // start calculations
            // top-N neighbors
            List<Map.Entry<Integer, Double>> sorted = Lists.sortMap(nns, true);
            int k = nns.size();
            if (k != 0) {
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
                    sum1 += (nns_ratings.get(ngbr) - userMeans.get(ngbr)) * en.getValue();
                }

                pred = userMeans.get(a) + sum1 / sum2;
            }

            return (pred > 0) ? pred : userMeans.get(a);
        }else{
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
    }

    protected SparseMatrix getUIMatrix(int ctx)
    {
        // Table {row-id, col-id, rate}
        Table<Integer, Integer, Double> dataTable_ui = HashBasedTable.create();

        // Map {col-id, multiple row-id}: used to fast build a rating matrix
        Multimap<Integer, Integer> colMap = HashMultimap.create();

        // read data to have a list of rating profiles for each uc pair
        SparseVector sv=trainMatrix.column(ctx);
        for(int ui: sv.getIndex()){
            int u = rateDao.getUserIdFromUI(ui);
            int j = rateDao.getItemIdFromUI(ui);
            dataTable_ui.put(u, j, sv.get(ui));
            colMap.put(j,u);
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
            this.sm=null;
            userCorrs=null;
            userMeans=null;

            carskit.data.structure.SparseMatrix UIM = getUIMatrix(ctx);
            this.sm=UIM;
            userCorrs = buildCorrs(true, UIM);
            userMeans = new DenseVector(numUsers);
            for (int u = 0; u < numUsers; u++) {
                SparseVector uv = UIM.row(u);
                userMeans.set(u, uv.getCount() > 0 ? uv.mean() : globalMean);
            }

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
    public String toString() {
        return Strings.toString(new Object[] { knn, similarityMeasure, similarityShrinkage });
    }

}
