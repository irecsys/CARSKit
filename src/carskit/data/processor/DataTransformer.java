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


import com.google.common.collect.*;
import org.apache.commons.lang3.StringUtils;
import com.sun.xml.internal.ws.api.pipe.FiberContextSwitchInterceptor;
import happy.coding.io.FileConfiger;
import happy.coding.io.FileIO;
import happy.coding.io.LineConfiger;
import happy.coding.io.Logs;
import happy.coding.io.Strings;
import happy.coding.io.net.EMailer;
import happy.coding.system.Dates;
import happy.coding.system.Systems;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Collection;

import librec.data.SparseMatrix;

public class DataTransformer implements Runnable {

    public DataTransformer() {
    }

    protected int flag;
    protected String dataPath;
    protected String outputfolder;

    public void setParameters(int f, String path, String folder) {
        this.flag = f;
        this.dataPath = path;
        this.outputfolder = folder;
    }

    public String TransformationFromLooseToBinary() throws Exception {
        BufferedReader br = FileIO.getReader(dataPath);
        String line = br.readLine(); // 1st line;
        HashMap<String, HashMap<String, String>> newlines = new LinkedHashMap();
        Multimap<String, String> conditions = TreeMultimap.create(); // key=dim, value=cond, keep the order when we adding to it
        while ((line = br.readLine()) != null) {
            String[] strs = line.split(",", -1);
            String key = strs[0].trim().toLowerCase() + "," + strs[1].trim().toLowerCase() + "," + strs[2].trim().toLowerCase(); // key = user,item,rating
            String cond = strs[4].trim().toLowerCase();
            if (cond.equals(""))
                cond = "na";
            conditions.put(strs[3].trim().toLowerCase(), cond);
            if (newlines.containsKey(key)) {
                HashMap<String, String> ratingcontext = newlines.get(key);
                ratingcontext.put(strs[3].trim().toLowerCase(), cond);

            } else {
                HashMap<String, String> ratingcontext = new HashMap();
                ratingcontext.put(strs[3].trim().toLowerCase(), cond);
                newlines.put(key, ratingcontext);
            }
        }
        br.close();

        this.PublishNewRatingFiles(outputfolder, conditions, newlines, true);

        if (FileIO.exist(outputfolder + "ratings_binary.txt"))
            return "Data transformaton completed (from Loose to Binary format). See new rating file: " + outputfolder + "ratings_binary.txt";
        else
            return "Data transformation failed. See output folder: " + outputfolder;
    }


    public String TransformationFromCompactToBinary() throws Exception {
        BufferedReader br = FileIO.getReader(dataPath);
        String line = br.readLine(); // 1st line;
        String[] header = line.split(",", -1);
        int dimscount = header.length - 3;
        String[] dims = new String[dimscount];
        for (int i = 3; i < header.length; ++i)
            dims[i - 3] = header[i].trim().toLowerCase();
        HashMap<String, HashMap<String, String>> newlines = new LinkedHashMap();
        Multimap<String, String> conditions = TreeMultimap.create(); // key=dim, value=cond, keep the order when we adding to it
        while ((line = br.readLine()) != null) {
            String[] strs = line.split(",", -1);
            HashMap<String, String> ratingcontext = new HashMap<>();
            for (int i = 3; i < 3 + dimscount; ++i) {
                String cond = strs[i].trim().toLowerCase();
                if (cond.equals(""))
                    cond = "na";
                ratingcontext.put(dims[i - 3], cond);
                conditions.put(dims[i - 3], cond);
            }
            newlines.put(line, ratingcontext); // the whole line is key
        }
        br.close();

        this.PublishNewRatingFiles(outputfolder, conditions, newlines, false);

        if (FileIO.exist(outputfolder + "ratings_binary.txt"))
            return "Data transformaton completed (from Compact to Binary format). See new rating file: " + outputfolder + "ratings_binary.txt";
        return "Data transformaton completed (from Compact to Binary format). See " + outputfolder;
    }

    private void PublishNewRatingFiles(String outputfolder, Multimap<String, String> conditions, HashMap<String, HashMap<String, String>> newlines, boolean isLoose) throws Exception {
        // add missing values to the condition sets
        for (String dim : conditions.keySet()) {
            conditions.put(dim, "na");
        }

        // create header
        StringBuilder headerBuilder = new StringBuilder();
        headerBuilder.append("user,item,rating");
        int start = 0;
        for (String dim : conditions.keySet()) {
            Collection<String> conds = conditions.get(dim);
            for (String cond : conds) {
                if (headerBuilder.length() > 0) headerBuilder.append(",");
                headerBuilder.append(dim + ":" + cond);
            }
        }
        String header = headerBuilder.toString();

        BufferedWriter bw = FileIO.getWriter(outputfolder + "ratings_binary.txt");
        bw.write(header + "\n");
        bw.flush();

        // start rewrite rating records
        for (String key : newlines.keySet()) {
            HashMap<String, String> ratingcontext = newlines.get(key);
            StringBuilder conditionBuilder = new StringBuilder();
            for (String dim : conditions.keySet()) {
                boolean isNA = false;
                boolean isCompleted = false;
                Collection<String> conds = conditions.get(dim);

                String dimCondition = ratingcontext.get(dim);
                if (dimCondition == null) {// 1st NA situation: because there is no such dim in this rating profile
                    isNA = true;
                } else if (dimCondition.equals("na")) // 2nd NA situation: it is already tagged with NA in this dim
                {
                    isNA = true;
                }

                for (String cond : conds) {
                    if (conditionBuilder.length() > 0) conditionBuilder.append(",");
                    if (isLoose) {
                        if (isNA) {
                            if (cond.equals("na")) {
                                conditionBuilder.append("1");
                                isCompleted = true;
                            } else
                                conditionBuilder.append("0");
                        } else {
                            if (isCompleted)
                                conditionBuilder.append("0");
                            else {
                                if (cond.equals(dimCondition)) {
                                    conditionBuilder.append("1");
                                    isCompleted = true; // have found one condition for this dimension, all others are 0
                                } else
                                    conditionBuilder.append("0");
                            }
                        }
                    } else {
                        if (dimCondition.equals(cond))
                            conditionBuilder.append("1");
                        else
                            conditionBuilder.append("0");
                    }
                }

            }
            String[] skey = key.split(",", -1); // when original format is compact, the key is whole line.
            if (skey.length > 3)
                key = skey[0].trim().toLowerCase() + "," + skey[1].trim().toLowerCase() + "," + skey[2].trim().toLowerCase();

            bw.write(key + "," + conditionBuilder.toString() + "\n");
            bw.flush();
        }
        bw.close();
    }

    @Override
    public void run() {
        try {
            switch (flag) {
                case 1: // it is binary format!
                    FileIO.copyFile(this.dataPath, this.outputfolder + "ratings_binary.txt");
                    break;
                case 2: // it is loose format!
                    Logs.warn("You rating data is in Loose format. CARSKit is working on transformation on the data format...");
                    Logs.info(this.TransformationFromLooseToBinary());
                    break;
                case 3: // it is compact format!
                    Logs.warn("You rating data is in Compact format. CARSKit is working on transformation on the data format...");
                    Logs.info(this.TransformationFromCompactToBinary());
                    break;
                default:
                    Logs.warn("You rating data is not shaped in the correct format. Please read our guideline on data preparation...");
                    break;
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
