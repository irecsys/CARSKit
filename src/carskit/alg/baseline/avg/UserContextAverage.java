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
import carskit.generic.Recommender;


/**
 * UserContextAverage: return average rating by a specific user in each context condition
 *
 * @author Yong Zheng
 *
 */


public class UserContextAverage extends Recommender {

    public UserContextAverage(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold){
        super(trainMatrix,testMatrix,fold);
        this.isCARSRecommender=true; // no 2D rating matrix will be created, i.e., this.train = null;
        this.algoName = "UserContextAvg";
    }

    @Override
    protected double predict(int u, int j, int c) throws Exception{

        double rate=rateDao.getUserContextAvg(trainMatrix, u, c);
        return (rate>0)?rate:this.globalMean;
    }
}
