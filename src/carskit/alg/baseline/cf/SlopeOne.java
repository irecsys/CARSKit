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

import happy.coding.io.Logs;
import librec.data.DenseMatrix;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import carskit.generic.Recommender;

/**
 * SlopeOne: Lemire, Daniel, and Anna Maclachlan. "Slope One Predictors for Online Rating-Based Collaborative Filtering." SDM. Vol. 5. 2005.
 * <p></p>
 * Note: This implementation is modified from the algorithm in LibRec-v1.3
 *
 * @author Yong Zheng
 *
 */

public class SlopeOne extends Recommender{

    private DenseMatrix devMatrix, cardMatrix;

    public SlopeOne(carskit.data.structure.SparseMatrix trainMatrix, carskit.data.structure.SparseMatrix testMatrix, int fold) {

        super(trainMatrix, testMatrix, fold);
        this.algoName = "SlopeOne";


    }

    @Override
    protected void initModel() throws Exception {
        super.initModel();
        devMatrix = new DenseMatrix(numItems,numItems);
        cardMatrix = new DenseMatrix(numItems,numItems);
    }

    @Override
    protected void buildModel() throws Exception {

        // compute items' differences
        for (int u = 0; u < numUsers; u++) {
            SparseVector uv = train.row(u);
            int[] items = uv.getIndex();

            for (int i : items) {
                double rui = uv.get(i);
                for (int j : items) {
                    if (i != j) {
                        double ruj = uv.get(j);
                        devMatrix.add(i, j, rui - ruj);
                        cardMatrix.add(i, j, 1);
                    }
                }
            }
        }

        // normalize differences
        for (int i = 0; i < numItems; i++) {
            for (int j = 0; j < numItems; j++) {
                double card = cardMatrix.get(i, j);
                if (card > 0) {
                    double sum = devMatrix.get(i, j);
                    devMatrix.set(i, j, sum / card);
                }
            }
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
    protected double predict(int u, int j) throws Exception{

        SparseVector uv = train.row(u, j);
        double preds = 0, cards = 0;
        for (int i : uv.getIndex()) {
            double card = cardMatrix.get(j, i);
            if (card > 0) {
                preds += (devMatrix.get(j, i) + uv.get(i)) * card;
                cards += card;
            }
        }

        return cards > 0 ? preds / cards : globalMean;
    }

}
