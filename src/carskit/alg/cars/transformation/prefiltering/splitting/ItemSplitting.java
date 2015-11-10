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
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.Multimap;
import com.google.common.collect.Table;
import com.google.common.primitives.Doubles;
import happy.coding.io.Logs;
import org.apache.commons.math3.stat.inference.TTest;

import java.util.Collection;
import java.util.List;


/**
 * ItemSplitting: Baltrunas, Linas, and Francesco Ricci. "Context-based splitting of item ratings in collaborative filtering." Proceedings of the third ACM conference on Recommender systems. ACM, 2009.
 * <p></p>
 * Note: t-test on the mean rating within two different contextual conditions is used to perform the splitting
 *
 * @author Yong Zheng
 *
 */


public class ItemSplitting implements ContextualSplitting {

    protected static Multimap<Integer, Integer> itemRatingList, condContextsList;
    protected static int startId;

    public Table<Integer, Integer, Integer> split(SparseMatrix sm, int min)
    {
        Table<Integer, Integer, Integer> datatable= HashBasedTable.create();

        for(Integer j : itemRatingList.keySet()){
            Collection<Integer> uis=itemRatingList.get(j);
            double maxt=Double.MIN_VALUE;
            int splitcond=-1;

            for(Integer cond : condContextsList.keySet()) {
                Collection<Integer> ctx = condContextsList.get(cond);
                // start to extract two rating list
                HashMultiset<Double> rate1 = HashMultiset.create();
                HashMultiset<Double> rate2 = HashMultiset.create();

                for (Integer ui : uis) {
                    List<Integer> uctx = sm.getColumns(ui);
                    for (Integer c : uctx) {
                        double rate = sm.get(ui, c);
                        if (ctx.contains(c))
                            rate1.add(rate);
                        else
                            rate2.add(rate);
                    }
                }

                double[] drate1 = Doubles.toArray(rate1);
                double[] drate2 = Doubles.toArray(rate2);

                if (drate1.length >= min && drate2.length >= min)
                {
                    TTest tt = new TTest();
                    double p = tt.tTest(drate1, drate2);
                    if (p < 0.05) {
                        double t = tt.t(drate1, drate2);
                        if (t > maxt) {
                            // update the split
                            splitcond = cond;
                            maxt = t;
                        }
                    }
                }
            }
            if(splitcond!=-1) {
                // put u, ctx, new uid into datatable
                int newid=startId++;
                Collection<Integer> ctx = condContextsList.get(splitcond);
                for(Integer c:ctx)
                    datatable.put(j,c,newid);
            }

        }
        Logs.info(datatable.rowKeySet().size() + " items have been splitted.");
        return datatable;
    }


    public ItemSplitting(int startId, Multimap<Integer, Integer> condContextsList, Multimap<Integer, Integer> itemRatingList)
    {
        ItemSplitting.startId =startId;
        ItemSplitting.itemRatingList =itemRatingList;
        ItemSplitting.condContextsList =condContextsList;
    }


}
