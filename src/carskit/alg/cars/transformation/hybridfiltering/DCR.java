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
import happy.coding.io.Lists;
import happy.coding.io.Logs;
import happy.coding.io.Strings;
import librec.data.*;

import java.util.*;

/**
 * Yong Zheng, Robin Burke, Bamshad Mobasher. "Optimal Feature Selection for Context-Aware Recommendation using Differential Relaxation". Proceedings of the 4th International Workshop on Context-Aware Recommender Systems, Dublin, Ireland, Sep 2012
 * <p></p>
 * MA Khanesar, M Teshnehlab, et al. "A novel binary particle swarm optimization." IEEE Mediterranean Conference on Control & Automation, 2007
 * <p></p>
 * Note: we just use three components as mentioned in the paper above, to reduce computational costs; we choose squared loss as fitness
 *
 * @author Yong Zheng
 *
 */

public class DCR extends IterativeRecommender {
    // user: nearest neighborhood
    private SymmMatrix userCorrs;
    private DenseVector userMeans;

    private int p; // number of particles
    private double lp; // particle learning rate
    private double lg; // global learning rate
    private double wt; // start weight
    private double wd; // end weight
    private double w;

    private int num_dim;
    private int num_component=3;

    private DenseVector pos_gbest; // the best historical particle position
    private double fitness_gbest; // the best historical fitness

    private Particle_BPSO[] swarm;
    private int len;
    private int start=-1;
    private String sol="";


    public DCR(carskit.data.structure.SparseMatrix trainMatrix, carskit.data.structure.SparseMatrix testMatrix, int fold) {

        super(trainMatrix, testMatrix, fold);
        this.algoName = "DCR";

        lp = algoOptions.getDouble("-lp");
        lg = algoOptions.getDouble("-lg");
        wt = algoOptions.getDouble("-wt");
        wd = algoOptions.getDouble("-wd");
        p = algoOptions.getInt("-p");
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
        swarm = new Particle_BPSO[p];
        for(int i=0;i<p;++i)
            swarm[i] = new Particle_BPSO(len);

    }

    @Override
    protected void buildModel() throws Exception {

        if(sol.equals("")) {
            for (int i = 0; i < p; ++i) { // for each particle
                Particle_BPSO bp = swarm[i];
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
                        for (int j = 0; j < len; ++j) {
                            double d11 = 0, d01 = 0, d12 = 0, d02 = 0;
                            double r1 = Math.random();
                            if (bp.pos.get(j) == 1) {
                                d11 = lp * r1;
                                d01 = 0 - d11;
                            } else {
                                d01 = lp * r1;
                                d11 = 0 - d01;
                            }
                            double r2 = Math.random();
                            if (pos_gbest.get(j) == 1) {
                                d12 = lg * r2;
                                d02 = 0 - d12;
                            } else {
                                d02 = lg * r2;
                                d12 = 0 - d02;
                            }

                            bp.volocity_1.set(j, w * bp.volocity_1.get(j) + d11 + d12);
                            bp.volocity_0.set(j, w * bp.volocity_0.get(j) + d01 + d02);

                            double v = 0;
                            if (bp.pos.get(j) == 0)
                                v = bp.volocity_1.get(j);
                            else
                                v = bp.volocity_0.get(j);
                            double sv = 1.0 / (1.0 + Math.exp(0 - v));
                            if (Math.random() < sv) {
                                if (bp.pos.get(j) == 1)
                                    bp.pos.set(j, 0);
                                else
                                    bp.pos.set(j, 1);
                            }
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
                    int bit = Integer.parseInt(strs[i].trim());
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
                if(ContextRelaxation(c, ctx, pos_3)){
                    part3 += rujc;
                    part3_count += 1.0;
                }
            }else{
                double sim = userCorrs.get(a, u);
                if(sim>0) {
                    // a potential neighbor
                    int j = rateDao.getItemIdFromUI(ui);
                    // double check whether this user has rated item t in c1
                    int newui = rateDao.getUserItemId(u + "," + t);
                    if (newui != -1) {
                        SparseVector sv = trainMatrix.row(newui);
                        if (sv != null) {
                            // search whether this user rated in c1
                            int[] cs = sv.getIndex();
                            double rate = ContextRelaxation(c, cs, pos_1, sv);

                            if (rate != -1) {
                                // this user is a successful neighbor
                                nns.put(u, sim);
                                // search whether this user rated in c2
                                rate = ContextRelaxation(c, cs, pos_2, sv);
                                if (rate == -1) {
                                    // if neighbor did not rate items in c2, return rate(u, t)
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
            List<Integer >uiids = trainMatrix.rows();
            for(int uiid:uiids){
                int u = rateDao.getUserIdFromUI(uiid);
                if (nns.containsKey(u)) {
                    SparseVector sv = trainMatrix.row(uiid);
                    int[] cs = sv.getIndex();
                    for (int ctx : cs) {
                        if (ContextRelaxation(c, ctx, pos_2)) {
                            double r = sv.get(ctx);
                            if (part22.containsKey(u)) {
                                part22.put(u, part22.get(u) + r);
                                part22_count.put(u, part22_count.get(u) + 1.0);
                            } else {
                                part22.put(u, r);
                                part22_count.put(u, 1.0);
                            }

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

    protected double ContextRelaxation(int c, int[] cs, double[] pos, SparseVector sv){
        double rate=-1;
        int index=-1;
        for(int ctx:cs){
            if(ContextRelaxation(c, ctx, pos)){
                index = ctx;
                break;
            }
        }
        if(index!=-1)
            rate = sv.get(index);
        return rate;
    }

    protected boolean ContextRelaxation(int c, int ctx, double[] pos){
        boolean mt = true;
        ArrayList<Integer> conds1=rateDao.getContextConditionsList().get(c);
        ArrayList<Integer> conds2=rateDao.getContextConditionsList().get(ctx);
        for(int i=0;i<pos.length;++i){
            if(pos[i]==1){
                if(conds1.get(i) != conds2.get(i)){
                    mt = false;
                    break;
                }
            }
        }
        return mt;
    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {

        return predict(u,j,c,this.pos_gbest);
    }

    @Override
    public String toString() {
        return Strings.toString(new Object[] { "p: "+p, "lp: "+lp, "lg: "+lg, "wt: "+wt, "wd: "+wd, "sol: "+pos_gbest.toString()});
    }

}
