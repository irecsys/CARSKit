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

package carskit.alg.cars.transformation.prefiltering.splitting;

import carskit.data.structure.SparseMatrix;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;

/**
 * UISplitting: Zheng, Yong, Robin Burke, and Bamshad Mobasher. "Splitting approaches for context-aware recommendation: An empirical study." Proceedings of the 29th Annual ACM Symposium on Applied Computing. ACM, 2014.
 * <p></p>
 * Note: t-test on the mean rating within two different contextual conditions is used to perform the splitting
 *
 * @author Yong Zheng
 *
 */

public class UISplitting implements ContextualSplitting {
    protected static Multimap<Integer, Integer> userRatingList, itemRatingList, condContextsList;
    protected static int startId_u, startId_i;

    public Table<Integer, Integer, Integer> splitUser(SparseMatrix sm, int min)
    {
        UserSplitting usp=new UserSplitting(startId_u,condContextsList, userRatingList);
        return usp.split(sm, min);
    }
    public Table<Integer, Integer, Integer> splitItem(SparseMatrix sm, int min)
    {
        ItemSplitting isp=new ItemSplitting(startId_i,condContextsList, itemRatingList);
        return isp.split(sm, min);
    }
    public Table<Integer, Integer, Integer> split(SparseMatrix sm, int min)
    {
        return null;
    }


    public UISplitting(int startIdu, int startIdi, Multimap<Integer, Integer> condContextsList,
                       Multimap<Integer, Integer> userRatingList, Multimap<Integer, Integer> itemRatingList)
    {
        startId_u=startIdu;
        startId_i=startIdi;
        UISplitting.userRatingList =userRatingList;
        UISplitting.itemRatingList =itemRatingList;
        UISplitting.condContextsList =condContextsList;
    }
}
