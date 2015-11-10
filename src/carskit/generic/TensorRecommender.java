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

package carskit.generic;

import java.util.*;

import carskit.data.structure.SparseMatrix;
import librec.data.SparseTensor;
import librec.data.Configuration;
import librec.data.MatrixEntry;
import librec.data.TensorEntry;
import happy.coding.io.Strings;
import happy.coding.io.FileIO;
import happy.coding.io.Logs;

/**
 * Interface for tensor recommenders
 *
 * @author Guo Guibing
 *
 */


@Configuration("factors, lRate, maxLRate, reg, iters, boldDriver")
public class TensorRecommender extends IterativeRecommender {

    /* for all tensors */
    protected static SparseTensor rateTensor;
    protected static int numDimensions, userDimension, itemDimension;
    protected static int[] dimensions;

    /* for a specific recommender */
    protected SparseTensor trainTensor, testTensor;

    static {
        rateTensor = rateDao.getRateTensor();
        numDimensions = rateTensor.numDimensions();
        dimensions = rateTensor.dimensions();

        userDimension = rateTensor.getUserDimension();
        itemDimension = rateTensor.getItemDimension();
    }

    public TensorRecommender(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) throws Exception {
        super(trainMatrix, testMatrix, fold);


        // construct train and test data
        trainTensor = rateTensor.clone();
        testTensor = new SparseTensor(dimensions);
        testTensor.setUserDimension(userDimension);
        testTensor.setItemDimension(itemDimension);

        for (MatrixEntry me : testMatrix) {
            int ui = me.row();
            int u=rateDao.getUserIdFromUI(ui);
            int i=rateDao.getItemIdFromUI(ui);

            List<Integer> indices = rateTensor.getIndices(u, i);

            for (int index : indices) {
                int[] keys = rateTensor.keys(index);
                testTensor.set(rateTensor.value(index), keys);
                trainTensor.remove(keys);
            }
        }
    }

    public int[] getKeys(int u, int i, int c){
        int[] keys = new int[numDimensions];
        keys[0] = u;
        keys[1] = i;
        HashMap<Integer, ArrayList<Integer>> dimensionConditionsList =rateDao.getDimensionConditionsList();
        ArrayList<Integer> conds=rateDao.getContextConditionsList().get(c);
        //System.out.println(Arrays.toString(conds.toArray()));
        int start=-1;
        for(int k=0;k<conds.size();++k){
            int condId = conds.get(k);
            int index = dimensionConditionsList.get(++start).indexOf(condId);
            if(index==-1)
                Logs.error("Index == -1: dimId = "+start+", condId = "+condId);
            keys[2+start] = index;
        }
        return keys;
    }


}
