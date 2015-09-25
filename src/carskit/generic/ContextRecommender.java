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

import carskit.data.structure.SparseMatrix;
import librec.data.DenseMatrix;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by yzheng on 8/5/15.
 */
public abstract class ContextRecommender extends IterativeRecommender {

    protected static int numConditions;
    protected static ArrayList<Integer> EmptyContextConditions;

    public ContextRecommender(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
        super(trainMatrix, testMatrix, fold);

        isCARSRecommender=true;

        numConditions = rateDao.numConditions();
        EmptyContextConditions = rateDao.getEmptyContextConditions();
    }

    @Override
    protected void initModel() throws Exception {

        super.initModel();

    }

    protected List<Integer> getConditions(int ctx)
    {
        String context=rateDao.getContextId(ctx);
        String[] cts = context.split(",");
        List<Integer> conds = new ArrayList<>();
        for(String ct:cts)
            conds.add(Integer.valueOf(ct));
        return conds;
    }
}
