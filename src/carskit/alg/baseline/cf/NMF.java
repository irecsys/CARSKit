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

package carskit.alg.baseline.cf;

import happy.coding.io.Strings;
import happy.coding.math.Randoms;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SparseVector;

import carskit.generic.IterativeRecommender;
import carskit.data.setting.Configuration;
import carskit.data.structure.SparseMatrix;

/**
 * NMF: Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative matrix factorization." Advances in neural information processing systems. 2001.<p></p>
 * Note: This implementation is modified from the algorithm in LibRec-v1.3
 *
 * @author Yong Zheng
 *
 */

@Configuration("factors, numIters")
public class NMF extends IterativeRecommender {

    // V = W * H
    protected DenseMatrix W, H;
    protected librec.data.SparseMatrix V;

    public NMF(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        // no need to update learning rate
        lRate = -1;
        this.algoName = "NMF";
    }

    @Override
    protected void initModel() throws Exception {
        super.initModel();
        W = new DenseMatrix(numUsers, numFactors);
        H = new DenseMatrix(numFactors, numItems);

        W.init(0.01);
        H.init(0.01);

        V = train;
    }

    @Override
    protected void buildModel() throws Exception {
        for (int iter = 1; iter <= numIters; iter++) {

            // update W by fixing H
            for (int u = 0; u < W.numRows(); u++) {
                SparseVector uv = V.row(u);

                if (uv.getCount() > 0) {
                    SparseVector euv = new SparseVector(V.numColumns());

                    for (int j : uv.getIndex())
                        euv.set(j, predict(u, j));

                    for (int f = 0; f < W.numColumns(); f++) {
                        DenseVector fv = H.row(f, false);
                        double real = fv.inner(uv);
                        double estm = fv.inner(euv) + 1e-9;

                        W.set(u, f, W.get(u, f) * (real / estm));
                    }
                }
            }

            // update H by fixing W
            DenseMatrix trW = W.transpose();
            for (int j = 0; j < H.numColumns(); j++) {
                SparseVector jv = V.column(j);

                if (jv.getCount() > 0) {
                    SparseVector ejv = new SparseVector(V.numRows());

                    for (int u : jv.getIndex())
                        ejv.set(u, predict(u, j));

                    for (int f = 0; f < H.numRows(); f++) {
                        DenseVector fv = trW.row(f, false);
                        double real = fv.inner(jv);
                        double estm = fv.inner(ejv) + 1e-9;

                        H.set(f, j, H.get(f, j) * (real / estm));
                    }
                }
            }

            // compute errors
            loss = 0;
            for (MatrixEntry me : V) {
                int u = me.row();
                int j = me.column();
                double ruj = me.get();

                if (ruj > 0) {
                    double euj = predict(u, j) - ruj;

                    loss += euj * euj;
                }
            }

            loss *= 0.5;

            if (isConverged(iter))
                break;
        }
    }

    @Override
    protected double predict(int u, int j, int c) throws Exception {
        if(isUserSplitting)
            u = userIdMapper.contains(u,c) ? userIdMapper.get(u,c) : u;
        if(isItemSplitting)
            j = itemIdMapper.contains(j,c) ? itemIdMapper.get(j,c) : j;
        return predict(u,j);
    }

    @Override
    protected double predict(int u, int j) throws Exception {
        return DenseMatrix.product(W, u, H, j);
    }

    @Override
    public String toString() {
        return Strings.toString(new Object[] { numFactors, numIters });
    }
}