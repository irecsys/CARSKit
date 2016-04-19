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

package carskit.alg.cars.adaptation.dependent;

import carskit.data.structure.DenseMatrix;
import carskit.data.structure.DenseVector;
import carskit.data.structure.SparseMatrix;
import carskit.generic.ContextRecommender;
import com.google.common.collect.HashBasedTable;
import librec.data.MatrixEntry;

import java.util.Iterator;

/**
 * FM (Factorization Machine): Rendle, Steffen, et al. "Fast context-aware recommendations with factorization machines." Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval. ACM, 2011.
 *
 * @author Yong Zheng
 *
 */


public class FM extends ContextRecommender {

    private double w0;
    private int p; // number of users + number of items + number of contextual conditions
    private int k; // number of factors
    private int size;
    private DenseVector w; // size = p
    private DenseMatrix V; // size = p x k
    private DenseMatrix Q; // size = n x k
    private float regLw, regLf;

    public FM(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);
        this.algoName = "FM";

        regLw = algoOptions.getFloat("-lw");
        regLf = algoOptions.getFloat("-lf");
    }

    @Override
    protected void initModel() throws Exception {
        super.initModel();

        k = numFactors;
        p = numUsers + numItems + numConditions;
        w0=0;
        size = trainMatrix.size();

        w = new DenseVector(p);
        w.init();

        V = new DenseMatrix(p,k);
        V.init(initMean, initStd);

        Q = new DenseMatrix(size, k);

    }

    private DenseVector getFeatureVector(int u, int j, int c)
    {
        DenseVector fs = new DenseVector(p);
        int indexu = u;
        int indexj = numUsers + j;
        int indexc = numUsers + numItems + c;
        for(int i=0;i<p;++i){
            if(i==indexu || i==indexj)
                fs.set(i,1);
            else if(i==indexc)
                fs.set(i,1.0/rateDao.numContextDims());
            else
                fs.set(i,0.0);
        }
        return fs;
    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        DenseVector fs = getFeatureVector(u,j,c);
        double pred=w0;
        for(j=0;j<p;++j)
            pred+=w.get(j)*fs.get(j);

        double sum = 0.0;
        for(int f=0;f<k;++f){
            double sum1=0.0, sum2=0.0;
            for(j=0;j<p;++j){
                double dot = V.get(j,f)*fs.get(j);
                sum1 += V.get(j,f)*fs.get(j);
                sum2 += Math.pow(dot, 2);
            }
            sum += Math.pow(sum1,2) - sum2;
        }

        pred+=0.5*sum;
        return pred;
    }

    @Override
    protected void buildModel() throws Exception {

        DenseVector errors=new DenseVector(size);
        Iterator<MatrixEntry> itor=trainMatrix.iterator();
        int counter=-1;
        HashBasedTable<Integer, Integer, Double> fvalues=HashBasedTable.create();
        while(itor.hasNext()){
            MatrixEntry me=itor.next();
            ++counter;
            int ui = me.row(); // user-item
            int u = rateDao.getUserIdFromUI(ui);
            int j = rateDao.getItemIdFromUI(ui);
            int c = me.column(); // context
            double rujc = me.get();

            DenseVector fs = getFeatureVector(u, j, c);

            double pred = predict(u, j, c);
            double euj = rujc - pred;

            errors.set(counter,euj);

            for(int f=0;f<k;++f) {
                double value=0;
                for(int i=0;i<p;++i) {
                    value += V.get(i, f) * fs.get(i);
                    fvalues.put(counter, i, fs.get(i));
                }
                Q.set(counter, f, value);
            }
        }

        for (int iter = 1; iter <= numIters; iter++) {

            loss=0;

            // update w0
            double update_w0=0;
            for(int i=0;i<size;++i) {
                double err=errors.get(i);
                update_w0 += err - w0;
                loss += err*err;
            }
            update_w0 = update_w0/(size + regLw);
            update_w0 = 0-update_w0;

            // update errors
            for(int i=0;i<size;++i)
                errors.set(i, errors.get(i) + update_w0-w0);

            loss += regLw*w0*w0;

            // copy new value to w0
            w0 = update_w0;

            // update vector w
            for(int l=0;l<p;++l){
                double update_wl=0;
                double sum=0;
                for(int i=0;i<size;++i){
                    double fl = fvalues.get(i,l);
                    update_wl += (errors.get(i) - w.get(l)*fl)*fl;
                    sum+= Math.pow(fl,2) + regLw;
                }
                update_wl = 0-update_wl/sum;

                // update error
                for(int i=0;i<size;++i)
                    errors.set(i, errors.get(i) + (update_wl - w.get(l))*fvalues.get(i,l));

                loss += regLw*w.get(l)*w.get(l);

                // copy new wl to w
                w.set(l, update_wl);

            }

            // update factors
            for(int f=0;f<k;++f)
                for(int l=0;l<p;++l){
                    // update Vlf
                    double update_Vlf=0;
                    double sum=0;
                    for(int i=0;i<size;++i){
                        double fl = fvalues.get(i,l);
                        double hlf = fl*Q.get(i,f) - Math.pow(fl,2)*V.get(l, f);
                        update_Vlf += (errors.get(i) - V.get(l,f)*hlf)*hlf;
                        sum += Math.pow(hlf,2) + regLf;

                        loss += regLf*Math.pow(Q.get(i,f),2);
                    }
                    update_Vlf = 0-update_Vlf/sum;

                    // update error and Q
                    for(int i=0;i<size;++i) {
                        errors.set(i, errors.get(i) + (update_Vlf - V.get(l, f)) * fvalues.get(i, l));
                        Q.set(i, f, Q.get(i, f) + (update_Vlf - V.get(l, f)) * fvalues.get(i, l));
                    }

                    // copy new Vlf to Vlf
                    V.set(l,f,update_Vlf);
                }
            loss *= 0.05;
        }
    }

}
