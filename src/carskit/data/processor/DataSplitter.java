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

package carskit.data.processor;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;

import carskit.data.structure.SparseMatrix;
import librec.data.SparseVector;

import happy.coding.io.FileIO;
import happy.coding.io.Lists;
import happy.coding.io.Logs;
import happy.coding.math.Randoms;
import happy.coding.math.Sortor;
import happy.coding.system.Debug;
import happy.coding.system.Systems;

/**
 * Created by yzheng on 7/31/15.
 */
public class DataSplitter {

    // [row-id, col-id, rate]
    private SparseMatrix rateMatrix;

    // [row-id, col-id, fold-id]
    private SparseMatrix assignMatrix;

    // number of folds
    private int numFold;

    public DataSplitter(SparseMatrix rateMatrix, int kfold){
        this.rateMatrix=rateMatrix;
        this.numFold=kfold;
        this.splitFolds(kfold);
    }

    public DataSplitter(SparseMatrix rateMatrix) {
        this.rateMatrix=rateMatrix;
    }

    /**
     * Return the k-th fold as test set (testMatrix), making all the others as train set in rateMatrix.
     *
     * @param k
     *            The index for desired fold.
     * @return Rating matrices {k-th train data, k-th test data}
     */
    public SparseMatrix[] getKthFold(int k) {
        if (k > numFold || k < 1)
            return null;

        SparseMatrix trainMatrix = new SparseMatrix(rateMatrix);
        SparseMatrix testMatrix = new SparseMatrix(rateMatrix);

        for (int u = 0, um = rateMatrix.numRows(); u < um; u++) {

            SparseVector items = rateMatrix.row(u);

            for (int j : items.getIndex()) {
                if (assignMatrix.get(u, j) == k)
                    trainMatrix.set(u, j, 0.0); // keep test data and remove train data
                else
                    testMatrix.set(u, j, 0.0); // keep train data and remove test data
            }
        }

        // remove zero entries
        SparseMatrix.reshape(trainMatrix);
        SparseMatrix.reshape(testMatrix);

        debugInfo(trainMatrix, testMatrix, k);

        return new SparseMatrix[] { trainMatrix, testMatrix };
    }

    /**
     * Split ratings into k-fold.
     *
     * @param kfold
     *            number of folds
     */
    private void splitFolds(int kfold) {
        assert kfold > 0;

        assignMatrix = new SparseMatrix(rateMatrix);

        int numRates = rateMatrix.getData().length;
        numFold = kfold > numRates ? numRates : kfold;

        // divide rating data into kfold sample of equal size
        double[] rdm = new double[numRates];
        int[] fold = new int[numRates];
        double indvCount = (numRates + 0.0) / numFold;

        for (int i = 0; i < numRates; i++) {
            rdm[i] = Randoms.uniform(); //Math.random();
            fold[i] = (int) (i / indvCount) + 1; // make sure that each fold has each size sample
        }

        Sortor.quickSort(rdm, fold, 0, numRates - 1, true);

        int[] row_ptr = rateMatrix.getRowPointers();
        int[] col_idx = rateMatrix.getColumnIndices();

        int f = 0;
        for (int u = 0, um = rateMatrix.numRows(); u < um; u++) {
            for (int idx = row_ptr[u], end = row_ptr[u + 1]; idx < end; idx++) {
                int j = col_idx[idx];
                // if randomly put an int 1-5 to entry (u, j), we cannot make sure equal size for each fold
                assignMatrix.set(u, j, fold[f++]);
            }
        }
    }

    /**
     * Split ratings into two parts: (ratio) training, (1-ratio) test subsets.
     *
     * @param ratio
     *            the ratio of training data over all the ratings.
     */
    public SparseMatrix[] getRatioByRating(double ratio) {

        assert (ratio > 0 && ratio < 1);

        SparseMatrix trainMatrix = new SparseMatrix(rateMatrix);
        SparseMatrix testMatrix = new SparseMatrix(rateMatrix);

        for (int u = 0, um = rateMatrix.numRows(); u < um; u++) {

            SparseVector uv = rateMatrix.row(u);
            for (int j : uv.getIndex()) {

                double rdm = Math.random();
                if (rdm < ratio)
                    testMatrix.set(u, j, 0.0);
                else
                    trainMatrix.set(u, j, 0.0);
            }
        }

        // remove zero entries
        SparseMatrix.reshape(trainMatrix);
        SparseMatrix.reshape(testMatrix);

        debugInfo(trainMatrix, testMatrix, -1);

        return new SparseMatrix[] { trainMatrix, testMatrix };
    }

    /**
     * print out debug information
     */
    private void debugInfo(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        String foldInfo = fold > 0 ? "Fold [" + fold + "]: " : "";
        Logs.debug("{}training amount: {}, test amount: {}", foldInfo, trainMatrix.size(), testMatrix.size());

        if (Debug.OFF) {
            String dir = Systems.getDesktop();
            try {
                FileIO.writeString(dir + "training.txt", trainMatrix.toString());
                FileIO.writeString(dir + "test.txt", testMatrix.toString());
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
