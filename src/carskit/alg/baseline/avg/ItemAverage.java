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


package carskit.alg.baseline.avg;

import carskit.data.structure.SparseMatrix;
import librec.data.SparseVector;
import carskit.generic.Recommender;

/**
 * ItemAverage: return average rating on a specific item
 *
 * @author Yong Zheng
 *
 */

public class ItemAverage extends Recommender {

    public ItemAverage(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold){
        super(trainMatrix,testMatrix,fold);

        this.algoName = "ItemAvg";
    }

    @Override
    protected double predict(int u, int j, int c) throws Exception{
        return predict(u,j);
    }

    @Override
    protected double predict(int u, int j) throws Exception{
        SparseVector sv=train.column(j);
        return (sv.getCount()>0) ? sv.mean() : this.globalMean;
    }
}
