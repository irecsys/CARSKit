package carskit.alg.cars.adaptation.dependent.sim;

import carskit.alg.cars.adaptation.dependent.CAMF;
import carskit.data.setting.Configuration;
import carskit.data.structure.SparseMatrix;
import carskit.generic.ContextRecommender;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.Table;
import happy.coding.io.Logs;
import happy.coding.math.Randoms;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SymmMatrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Yong Zheng, Bamshad Mobasher, Robin Burke. "Similarity-Based Context-aware Recommendation", Proceedings of the 16th International Conference on Web Information System Engineering (WISE), pp. 431-447
 */

// To guarantee the distance value is in range [0,1], it is better to use projected or bounded gradient method, see "Projected Gradient Methods for Non-negative Matrix Factorization"
// In this paper, we use a simple rule as constraint to limit the distance value.


public class CAMF_MCS extends CAMF{

    private double upbound;
    private double lowbound;

    public CAMF_MCS(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);
        this.algoName = "CAMF_MCS";
        // it is an algorithm for top-N recommendations, since the predicted score is used to rank the items. The predicted score is not guaranteed to stay in the original rating scale.
        isRankingPred = true;
    }

    protected void initModel() throws Exception {

        super.initModel();


        upbound = 1.0/Math.sqrt(rateDao.numContextDims());
        lowbound = 1.0/Math.pow(10, 100);

        cVector_MCS=new DenseVector(numConditions);
        cVector_MCS.init(upbound);

    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        double pred=DenseMatrix.rowMult(P, u, Q, j);
        List<Integer> conditions=getConditions(c);
        double dist=0;
        for(int i=0;i<conditions.size();++i) {
            int index1=conditions.get(i);
            int index2=EmptyContextConditions.get(i);
            dist += Math.pow(cVector_MCS.get(index1) - cVector_MCS.get(index2), 2);
        }
        dist = Math.sqrt(dist);
        double sim=1-dist;
        //sim = (sim>1)?upbound:sim;
        //sim = (sim<0)?lowbound:sim;
        pred = pred*sim;
        return pred;
    }

    @Override
    protected void buildModel() throws Exception {

        for (int iter = 1; iter <= numIters; iter++) {

            loss = 0;
            for (MatrixEntry me : trainMatrix) {

                int ui = me.row(); // user-item
                int u= rateDao.getUserIdFromUI(ui);
                int j= rateDao.getItemIdFromUI(ui);
                int ctx = me.column(); // context
                double rujc = me.get();

                HashBasedTable<Integer, Integer, Double> toBeUpdated = HashBasedTable.create();
                double simc=1.0;
                double dotRating=DenseMatrix.rowMult(P, u, Q, j);

                double pred=dotRating;
                List<Integer> conditions=getConditions(ctx);
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

                double sim=1-dist;
                //sim = (sim>1)?upbound:sim;
                //sim = (sim<0)?lowbound:sim;
                pred*=sim;


                double euj = rujc - pred;


                loss += euj * euj;

                // update similarity values
                if(toBeUpdated.size()>0) {
                    for (int index1 : toBeUpdated.rowKeySet())
                        for (int index2 : toBeUpdated.row(index1).keySet()) {
                            double pos1 = cVector_MCS.get(index1);
                            double pos2 = cVector_MCS.get(index2);

                            if(dist==0)
                                dist=lowbound;

                            double pos1_update = pos1 + lRate*(euj*dotRating*toBeUpdated.get(index1,index2)/dist - regC*pos1);
                            double pos2_update = pos2 - lRate*(euj*dotRating*toBeUpdated.get(index1,index2)/dist + regC*pos2);

                            // In this paper, we use a simple rule as constraint to limit the distance value.

                            pos1_update = (pos1_update<0)?lowbound:pos1_update;
                            pos1_update = (pos1_update>upbound)?upbound-lowbound:pos1_update;

                            pos2_update = (pos2_update<0)?lowbound:pos2_update;
                            pos2_update = (pos2_update>upbound)?upbound-lowbound:pos2_update;

                            cVector_MCS.set(index1,pos1_update);
                            cVector_MCS.set(index2,pos2_update);
                        }
                }

                // update user and item vectors

                for (int f = 0; f < numFactors; f++) {
                    double puf = P.get(u, f);
                    double qjf = Q.get(j, f);

                    double delta_u = euj * qjf * (1-dist)- regU * puf;
                    double delta_j = euj * puf * (1-dist) - regI * qjf;

                    P.add(u, f, lRate * delta_u);
                    Q.add(j, f, lRate * delta_j);

                    loss += regU * puf * puf + regI * qjf * qjf;
                }

            }
            loss *= 0.05;

            if (isConverged(iter))
                break;

        }// end of training

    }
}
