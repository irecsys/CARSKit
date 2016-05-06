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

package carskit.alg.cars.transformation.hybridfiltering;

import carskit.generic.IterativeRecommender;
import com.sun.deploy.util.ArrayUtil;
import happy.coding.io.Lists;
import happy.coding.io.Logs;
import happy.coding.io.Strings;
import happy.coding.math.Stats;
import librec.data.*;
import org.apache.commons.lang3.ArrayUtils;

import java.util.*;

/**
 * Yong Zheng, Robin Burke, Bamshad Mobasher. "Recommendation with differential context weighting." User Modeling, Adaptation, and Personalization. Springer Berlin Heidelberg, 2013. 152-164.
 * <p></p>
 * M Clerc, J Kennedy. "The particle swarm-explosion, stability, and convergence in a multidimensional complex space." IEEE Transactions on Evolutionary Computation, 6.1 (2002): 58-73.
 * <p></p>
 * Note: we just use three components as mentioned in the paper above, to reduce computational costs; we choose squared loss as fitness; we set a global threshold for all components
 *
 * @author Yong Zheng
 *
 */

public class DCW extends IterativeRecommender {
    // user: nearest neighborhood
    private SymmMatrix userCorrs;
    private DenseVector userMeans;

    private int p; // number of particles
    private double lp; // particle learning rate
    private double lg; // global learning rate
    private double wt; // start weight
    private double wd; // end weight
    private double w;
    private double th;

    private int num_dim;
    private int num_component=3;
    double p1=3, p2=4;

    private DenseVector pos_gbest; // the best historical particle position
    private double fitness_gbest; // the best historical fitness

    private Particle_CFPSO[] swarm;
    private int len;
    private int start=-1;
    private String sol="";


    public DCW(carskit.data.structure.SparseMatrix trainMatrix, carskit.data.structure.SparseMatrix testMatrix, int fold) {

        super(trainMatrix, testMatrix, fold);
        this.algoName = "DCW";

        lp = algoOptions.getDouble("-lp");
        lg = algoOptions.getDouble("-lg");
        wt = algoOptions.getDouble("-wt");
        wd = algoOptions.getDouble("-wd");
        p = algoOptions.getInt("-p");
        th = algoOptions.getDouble("-th");
        sol = algoOptions.getString("-sol","");
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
        num_dim=rateDao.numContextDims();

        fitness_gbest = Double.MAX_VALUE;
        this.len = num_dim*num_component;
        pos_gbest = new DenseVector(len);
        swarm = new Particle_CFPSO[p];
        for(int i=0;i<p;++i)
            swarm[i] = new Particle_CFPSO(len);

    }

    @Override
    protected void buildModel() throws Exception {

        if(sol.equals("")) {
            for (int i = 0; i < p; ++i) { // for each particle
                Particle_CFPSO bp = swarm[i];
                // let this particle run a number of iterations
                for (int iter = 1; iter <= numIters; iter++) { // for each iteration
                    double loss = 0;
                    for (MatrixEntry me : trainMatrix) {
                        int ui = me.row(); // user-item
                        int u = rateDao.getUserIdFromUI(ui);
                        int j = rateDao.getItemIdFromUI(ui);
                        int ctx = me.column(); // context
                        double rujc = me.get(); // real rating
                        double predication = predict(u, j, ctx, bp.pos);
                        loss += Math.pow((rujc - predication), 2);
                    }

                    if (start == -1) {
                        start = 0;
                        // update global best position and fitness
                        if (loss < fitness_gbest) {
                            fitness_gbest = loss;
                            pos_gbest = bp.pos.clone();
                        }
                    }

                    DenseVector tmp = bp.pos.clone();

                    // update partcile best position and fitness
                    if (loss < bp.fitness_best) {
                        bp.fitness_best = loss;
                        bp.pos_best = bp.pos.clone();

                        // update its position for the next iteration
                        w = wd + (wt - wd) * (numIters - iter) / numIters;
                        double x = 2 * Math.random() / Math.abs(2 - lp - lg - Math.sqrt((lp + lg) * (lp + lg - 4)));
                        for (int j = 0; j < len; ++j) {
                            bp.volocity.set(j, bp.volocity.get(j) + lp * (p1 - bp.pos.get(j) + lg * (p2 - bp.pos.get(j))));
                            bp.pos.set(j, x * bp.volocity.get(j) + x * bp.pos.get(j) + (1 - x) * (lp * p1 + lg * p2) / (lp + lg));
                        }
                    }

                    // update global best position and fitness
                    if (loss < fitness_gbest) {
                        fitness_gbest = loss;
                        pos_gbest = tmp;
                    }

                    Logs.info("Fold[" + fold + "]: current particle: " + (i + 1) + ", current iteration: " + iter + ", current loss: " + loss + ", lowest loss: " + fitness_gbest);
                }
            }
        }else{
            // load solution to memory
            String[] strs=sol.split(";",-1);
            if(strs.length!=this.len){
                Logs.error("Error: the length of your solution should be "+this.len);
                return;
            }else{
                for(int i=0;i<this.len;++i){
                    double bit = Double.parseDouble(strs[i].trim());
                    pos_gbest.set(i, bit);
                }
                Logs.info("You solution has been successfully loaded.");
            }
        }
    }

    protected double predict(int a, int t, int c, DenseVector position) throws Exception {
        double pred = 0;
        double[] pos = position.getData();
        double[] pos_1 = new double[num_dim];
        double[] pos_2 = new double[num_dim];
        double[] pos_3 = new double[num_dim];

        for(int i=0;i<pos.length;++i){
            if(i<num_dim)
                pos_1[i] = pos[i];
            else if(i<2*num_dim)
                pos_2[i-num_dim] = pos[i];
            else
                pos_3[i-2*num_dim] = pos[i];
        }

        double part3=0, part3_count=0;
        HashMap<Integer, Double> part22 =new  HashMap<Integer, Double>();
        HashMap<Integer, Double> part22_count =new  HashMap<Integer, Double>();
        HashMap<Integer, Double> part21 = new HashMap<>();

        HashMap<Integer, Double> nns = new HashMap<>(); // key = ngbr id, value = sim

        for (MatrixEntry me : trainMatrix) {
            int ui = me.row(); // user-item
            int u= rateDao.getUserIdFromUI(ui);
            int ctx = me.column(); // context
            double rujc = me.get(); // real rating
            if(u == a){
                double sim = ContextSimilarity(c, ctx, pos_3);
                if(sim>=th) {
                    part3 += sim * rujc;
                    part3_count += sim;
                }
            } else{
                double simu = userCorrs.get(a, u);
                if(simu>0) {
                    // a potential neighbor
                    int j = rateDao.getItemIdFromUI(ui);
                    // double check whether this user has rated item t in c1
                    int newui = rateDao.getUserItemId(u + "," + t);
                    if (newui != -1) {
                        SparseVector sv = trainMatrix.row(newui);
                        if (sv != null) {
                            // user has rated item t in some contexts
                            int[] cs = sv.getIndex();
                            if (ContextMatch(c, cs, pos_1)){
                                // this user is a successful neighbor
                                nns.put(u, simu);
                                // get value for part21
                                double rate = ContextWeight(c, cs, pos_2, sv);
                                if (rate == -1) {
                                    // if not rating profiles with a simlarity larger than threshold
                                    rate = train.get(u, t);
                                }
                                part21.put(u, rate);
                            }
                        }
                    }
                }

            }
        }

        // get user average
        if(part3_count==0)
            part3 = userMeans.get(a);
        else
            part3/=part3_count;

        pred+=part3;

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


            // calculate part22 first
            List<Integer> uiids=trainMatrix.rows();
            for(int uiid:uiids){
                int u = rateDao.getUserIdFromUI(uiid);
                if (nns.containsKey(u)) {
                    SparseVector sv = trainMatrix.row(uiid);
                    int[] cs = sv.getIndex();
                    for (int ctx : cs) {
                        double sim = ContextSimilarity(c, ctx, pos_2);
                        double r = sv.get(ctx);
                        if (part22.containsKey(u)) {
                            part22.put(u, part22.get(u) + sim*r);
                            part22_count.put(u, part22_count.get(u) + sim);
                        } else {
                            part22.put(u, sim*r);
                            part22_count.put(u, sim);
                        }
                    }
                }
            }


            double sum1 = 0;
            double sum2 = 0;

            for (Map.Entry<Integer, Double> en : nns.entrySet()) {
                int ngbr = en.getKey();
                sum2 += en.getValue();

                double tmp = 0;
                if (part22.containsKey(ngbr))
                    tmp = part22.get(ngbr) / part22_count.get(ngbr);
                else
                    tmp = userMeans.get(ngbr);

                sum1 += en.getValue() * (part21.get(ngbr) - tmp);
            }

            pred += sum1 / sum2;
        }

        return pred;
    }

    protected double ContextWeight(int c, int[] cs, double[] pos, SparseVector sv){
        double rate=0, count=0;

        for(int ctx:cs){
            double sim=ContextSimilarity(c, ctx, pos);
            if(sim>=th){
                rate+=sv.get(ctx);
                count+=1.0;
            }
        }
        if(count==0)
            return -1;
        else
            return rate/count;
    }

    protected boolean ContextMatch(int c, int[] cs, double[] pos){
        boolean okay=false;
        for(int ctx:cs){
            double sim=ContextSimilarity(c, ctx, pos);
            if(sim>=th){
                okay=true;
                break;
            }
        }
        return okay;
    }

    protected double ContextSimilarity(int c, int ctx, double[] pos){
        double sim=0;
        ArrayList<Integer> conds1=rateDao.getContextConditionsList().get(c);
        ArrayList<Integer> conds2=rateDao.getContextConditionsList().get(ctx);
        for(int i=0;i<pos.length;++i){
                if(conds1.get(i) == conds2.get(i))
                    sim+=pos[i];
        }
        return sim/ Stats.sum(pos);
    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {

        return predict(u,j,c,this.pos_gbest);
    }

    @Override
    public String toString() {
        return Strings.toString(new Object[] { "p: "+p, "lp: "+lp, "lg: "+lg, "wt: "+wt, "wd: "+wd, "th: "+th, "sol: "+pos_gbest.toString()});
    }

}
