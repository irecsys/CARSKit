package carskit.alg.cars.adaptation.independent;

import carskit.generic.Recommender;
import com.google.common.collect.HashMultimap;
import happy.coding.io.Lists;
import happy.coding.io.Strings;
import librec.data.DenseVector;
import librec.data.SparseVector;
import librec.data.SymmMatrix;

import java.util.*;

/**
 * Implement the method in: Chen, Annie. "Context-aware collaborative filtering system: Predicting the user’s preference in the ubiquitous computing environment." International Symposium on Location-and Context-Awareness. Springer, Berlin, Heidelberg, 2005.
 * We use equation 5) in the paper as the rating prediction function
 */

public class Chen1 extends Recommender {

    // user: nearest neighborhood
    private SymmMatrix userCorrs;
    private DenseVector userMeans;
    HashMultimap<Integer, HashMap<String, Double>> item_ContextsSimilarity=HashMultimap.create();


    public Chen1(carskit.data.structure.SparseMatrix trainMatrix, carskit.data.structure.SparseMatrix testMatrix, int fold) {

        super(trainMatrix, testMatrix, fold);
        this.algoName = "Chen1";

    }


    @Override
    protected void initModel() throws Exception {
        super.initModel();
        /*
        int[] c=trainMatrix.getColumnIndices();
        int numdims=rateDao.getContextIds().size();
        Set<Integer> items=rateDao.getItemIds().values();
        Set<Integer> users=rateDao.getUserIds().values();

        HashMap<Integer, ArrayList<Integer>> ctxConditions=rateDao.getContextConditionsList();
        HashMultimap<Integer, Integer> condContexts=(HashMultimap)rateDao.getConditionContextsList();

        for(Integer item:items){
            // get average item rating
            SparseVector sv=train.column(item);
            double item_avg=(sv.getCount()>0) ? sv.mean() : this.globalMean;
            for(int i=0;i<c.length;++i){
                int c1=c[i];
                ArrayList<Integer> c1_conditions=ctxConditions.get(c1);
                for(int j=i+1;j<c.length;++j){
                    int c2=c[j];
                    ArrayList<Integer> c2_conditions=ctxConditions.get(c2);

                    // context similarity = summation of similarity at each dimension or bit
                    double sim=0, count=0;
                    for(int k=0;k<c1_conditions.size();++k){
                        // calculate similarity bit by bit
                        int condk_c1=c1_conditions.get(k);
                        int condk_c2=c2_conditions.get(k);

                        Set<Integer> ctx1=condContexts.get(condk_c1);
                        Set<Integer> ctx2=condContexts.get(condk_c2);

                        ArrayList<Double> component1=new ArrayList<>();
                        ArrayList<Double> component2=new ArrayList<>();

                        for(Integer user:users){
                            int ui=rateDao.getUserItemId(user+","+item);
                            if(ui==-1 || !trainMatrix.rows().contains(ui))
                                continue;
                            double rate_ui_condk1=0;
                            double count1=0;
                            double rate_ui_condk2=0;
                            double count2=0;

                            for(Integer ccc:ctx1){
                                double rate=trainMatrix.get(ui,ccc);
                                if(rate>0){
                                    rate_ui_condk1+=rate;
                                    count1+=1.0;
                                }
                            }
                            for(Integer ccc:ctx2){
                                double rate=trainMatrix.get(ui,ccc);
                                if(rate>0){
                                    rate_ui_condk2+=rate;
                                    count2+=1.0;
                                }
                            }

                            if(count1==0 || count2==0){
                                continue;
                            }else{
                                rate_ui_condk1/=count1;
                                rate_ui_condk2/=count2;
                                component1.add(rate_ui_condk1-item_avg);
                                component2.add(rate_ui_condk2-item_avg);
                            }
                        }
                        // start calculating similarity on the current bit
                        double sim1=0, sim2=0, sim3=0;
                        for(int s=0;s<component1.size();++s){
                            double a=component1.get(s), b=component2.get(s);
                            sim1+=a*b;
                            sim2+=a*a;
                            sim3+=b*b;
                        }
                        double simt=(Math.sqrt(sim2)*Math.sqrt(sim3));
                        if(simt!=0){
                            simt=sim1/simt;
                            sim+=simt;
                            count+=1.0;
                        }
                    }
                    if(count>0) {
                        sim = sim / count;
                        HashMap<String, Double> map=new HashMap<>();
                        map.put(c1+","+c2,sim);
                        map.put(c2+","+c1,sim);
                        item_ContextsSimilarity.put(item, map);
                    }
                }
            }
        }
        */
    }

    protected double getContextsSimilarity(int item, int c1, int c2) throws Exception{
        HashMap<Integer, ArrayList<Integer>> ctxConditions=rateDao.getContextConditionsList();
        HashMultimap<Integer, Integer> condContexts=(HashMultimap)rateDao.getConditionContextsList();
        ArrayList<Integer> c1_conditions=ctxConditions.get(c1);
        ArrayList<Integer> c2_conditions=ctxConditions.get(c2);
        List<Integer> users=train.columns();

        // get average item rating
        SparseVector sv=train.column(item);
        double item_avg=(sv.getCount()>0) ? sv.mean() : this.globalMean;

        // context similarity = summation of similarity at each dimension or bit
        double sim=0, count=0;
        for(int k=0;k<c1_conditions.size();++k){
            // calculate similarity bit by bit
            int condk_c1=c1_conditions.get(k);
            int condk_c2=c2_conditions.get(k);

            Set<Integer> ctx1=condContexts.get(condk_c1);
            Set<Integer> ctx2=condContexts.get(condk_c2);

            ArrayList<Double> component1=new ArrayList<>();
            ArrayList<Double> component2=new ArrayList<>();

            for(Integer user:users){
                int ui=rateDao.getUserItemId(user+","+item);
                if(ui==-1 || !trainMatrix.rows().contains(ui))
                    continue;
                double rate_ui_condk1=0;
                double count1=0;
                double rate_ui_condk2=0;
                double count2=0;

                for(Integer ccc:ctx1){
                    double rate=trainMatrix.get(ui,ccc);
                    if(rate>0){
                        rate_ui_condk1+=rate;
                        count1+=1.0;
                    }
                }
                for(Integer ccc:ctx2){
                    double rate=trainMatrix.get(ui,ccc);
                    if(rate>0){
                        rate_ui_condk2+=rate;
                        count2+=1.0;
                    }
                }

                if(count1==0 || count2==0){
                    continue;
                }else{
                    rate_ui_condk1/=count1;
                    rate_ui_condk2/=count2;
                    component1.add(rate_ui_condk1-item_avg);
                    component2.add(rate_ui_condk2-item_avg);
                }
            }
            // start calculating similarity on the current bit
            double sim1=0, sim2=0, sim3=0;
            for(int s=0;s<component1.size();++s){
                double a=component1.get(s), b=component2.get(s);
                sim1+=a*b;
                sim2+=a*a;
                sim3+=b*b;
            }
            double simt=(Math.sqrt(sim2)*Math.sqrt(sim3));
            if(simt!=0){
                simt=sim1/simt;
                sim+=simt;
                count+=1.0;
            }
        }
        return (count==0)?sim:sim/count;
    }



    @Override
    protected double predict(int u, int j, int c) throws Exception {
        double rate=0;
        int ui=rateDao.getUserItemId(u+","+j);
        if(!trainMatrix.rows().contains(ui)){
            rate=train.get(u,j);
            rate=(rate>0)?rate:this.globalMean;
        }
        else {
            List<Integer> ratedContexts = trainMatrix.getColumns(ui);
            if (ratedContexts.size() == 0) {
                // use avg(u, j) as predicted rating
                rate = train.get(u, j);
                rate = (rate > 0) ? rate : this.globalMean;
            } else {
                double d1 = 0, d2 = 0;
                for (int k = 0; k < ratedContexts.size(); ++k) {
                    int ratedContext=ratedContexts.get(k);
                    double r = trainMatrix.get(ui, ratedContext);
                    double sim = 0;

                    Set<HashMap<String, Double>> ctxSimilarity = item_ContextsSimilarity.get(j);
                    String key = c + "," + ratedContext;
                    boolean exist=false;
                    for (HashMap<String, Double> entry : ctxSimilarity) {
                        if (entry.containsKey(key)) {
                            sim = entry.get(key);
                            exist=true;
                            break;
                        }
                    }
                    if(!exist){
                        sim=this.getContextsSimilarity(j,c,ratedContext);
                        HashMap<String, Double> map=new HashMap<>();
                        map.put(key,sim);
                        map.put(ratedContext+","+c,sim);
                        item_ContextsSimilarity.put(j,map);
                    }

                    d1 += r * sim;
                    d2 += sim;
                }

                if (d2 == 0) {
                    rate = train.get(u, j);
                    rate = (rate > 0) ? rate : this.globalMean;
                } else {
                    rate = d1 / d2;
                }
            }
        }
        return rate;
    }


    @Override
    public String toString() {
        return Strings.toString(new Object[] { knn, similarityMeasure, similarityShrinkage });
    }

}
