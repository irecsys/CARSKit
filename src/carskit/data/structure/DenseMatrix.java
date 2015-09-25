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

package carskit.data.structure;

/**
 * Created by yzheng on 7/31/15.
 */
public class DenseMatrix extends librec.data.DenseMatrix {
    public DenseMatrix(int numRows, int numColumns) {
        super(numRows, numColumns);
    }

    public DenseMatrix(double[][] array) {
        super(array);
    }

    public DenseMatrix(librec.data.DenseMatrix mat) {
        super(mat);
    }
}
