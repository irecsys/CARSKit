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


public class CAMF_LCS extends CAMF{

    private int numF;

    public CAMF_LCS(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);
        this.algoName = "CAMF_LCS";
    }

    protected void initModel() throws Exception {

        super.initModel();
        numF=algoOptions.getInt("-f", 10);

        cfMatrix_LCS=new DenseMatrix(numConditions, numF);
        cfMatrix_LCS.init();
    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        double pred=DenseMatrix.rowMult(P, u, Q, j);
        List<Integer> conditions=getConditions(c);
        for(int i=0;i<conditions.size();++i){
            double[] dv1=cfMatrix_LCS.row(conditions.get(i)).getData();
            double[] dv2=cfMatrix_LCS.row(EmptyContextConditions.get(i)).getData();
            double sum1=0,sum2=0;
            for(int h=0;h<dv1.length;++h){
                sum1+=dv1[h]*dv1[h];
                sum2+=dv2[h]*dv2[h];
            }
            sum1=Math.sqrt(sum1);
            sum2=Math.sqrt(sum2);
            //if(isRankingPred)
            pred=pred*DenseMatrix.rowMult(cfMatrix_LCS, conditions.get(i), cfMatrix_LCS, EmptyContextConditions.get(i));
            //else
            //pred=pred*DenseMatrix.rowMult(cfMatrix_LCS, conditions.get(i), cfMatrix_LCS, EmptyContextConditions.get(i))/(sum1*sum2);
        }
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
                for(int i=0;i<conditions.size();++i) {
                    int index1=conditions.get(i);
                    int index2=EmptyContextConditions.get(i);
                    double sim=1.0;
                    if(index1!=index2) {
                        sim = DenseMatrix.rowMult(cfMatrix_LCS, index1, cfMatrix_LCS, index2);
                        toBeUpdated.put(index1,index2,sim);
                        simc*=sim;
                    }
                    //loss += regC * sim * sim;
                    pred = pred * sim;
                }

                double euj = rujc - pred;

                loss += euj * euj;

                // update similarity values
                // in LCS, it is to update the vector representations
                if(toBeUpdated.size()>0) {
                    for (int index1 : toBeUpdated.rowKeySet())
                        for (int index2 : toBeUpdated.row(index1).keySet()) {
                            // index1 and index2 are pairwise
                            // for each index2, there is only one index2
                            for (int f = 0; f < numF; f++) {
                                double c1f = cfMatrix_LCS.get(index1,f);
                                double c2f = cfMatrix_LCS.get(index2,f);
                                double sim = toBeUpdated.get(index1,index2);
                                double delta_c1 = euj*dotRating*simc*c2f/sim - regC*c1f;
                                double delta_c2 = euj*dotRating*simc*c1f/sim - regC*c2f;

                                cfMatrix_LCS.add(index1,f,lRate*delta_c1);
                                cfMatrix_LCS.add(index2,f,lRate*delta_c2);

                                loss += regC* c1f * c1f + regC * c2f * c2f;
                            }
                        }
                }

                // update user and item vectors

                for (int f = 0; f < numFactors; f++) {
                    double puf = P.get(u, f);
                    double qjf = Q.get(j, f);

                    double delta_u = euj * qjf * simc- regU * puf;
                    double delta_j = euj * puf * simc - regI * qjf;

                    P.add(u, f, lRate * delta_u);
                    Q.add(j, f, lRate * delta_j);

                    loss += regU * puf * puf + regI * qjf * qjf;
                }

            }
            loss *= 0.5;

            if (isConverged(iter))
                break;

        }// end of training

    }
}
