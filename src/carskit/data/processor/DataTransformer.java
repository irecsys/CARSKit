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
import java.util.*;

import librec.data.SparseMatrix;

public class DataTransformer implements Runnable {

    public DataTransformer() {
    }

    protected int flag_train, flag_test;
    protected String dataPath_train, dataPath_test;
    protected String outputfolder;

    public void setParameters(int f_train, String path_train, int f_test, String path_test, String folder) {
        this.flag_train = f_train;
        this.flag_test = f_test;
        this.dataPath_train = path_train;
        this.dataPath_test = path_test;
        this.outputfolder = folder;
    }

    private Multimap<String, String> getConditions()throws Exception{
        //Multimap<String, String> conditions=LinkedHashMultimap.create();
        Logs.info("flags - train: "+flag_train+", test: "+flag_test);
        Multimap<String, String> conditions=TreeMultimap.create();
        switch(flag_train){
            case 1:
                getConditionsFromBinaryData(dataPath_train, conditions);
                break;
            case 2:
                getConditionsFromLooseData(dataPath_train, conditions);
                break;
            case 3:
                getConditionsFromCompactData(dataPath_train, conditions);
                break;
        }
        switch(flag_test){
            case 1:
                getConditionsFromBinaryData(dataPath_test, conditions);
                break;
            case 2:
                getConditionsFromLooseData(dataPath_test, conditions);
                break;
            case 3:
                getConditionsFromCompactData(dataPath_test, conditions);
                break;
            case -1:
                break;
        }

        //Iterate and add "na" conditions
        for(String dim:conditions.keySet()){
            Collection<String> conds=conditions.get(dim);
            if(!conds.contains("na"))
                conditions.put(dim,"na");
        }
        return conditions;
    }

    private void getConditionsFromBinaryData(String dataPath, Multimap<String, String> conditions) throws Exception{
        BufferedReader br = FileIO.getReader(dataPath);
        String line = br.readLine(); // 1st line;
        br.close();
        String[] header = line.split(",", -1);
        for (int i = 3; i < header.length; ++i){
            String[] strs=header[i].split(":",-1);
            conditions.put(strs[0].trim().toLowerCase(),strs[1].trim().toLowerCase());
        }
    }

    private void getConditionsFromLooseData(String dataPath, Multimap<String, String> conditions) throws Exception{
        BufferedReader br = FileIO.getReader(dataPath);
        String line = br.readLine(); // 1st line;
        while ((line = br.readLine()) != null) {
            String[] strs = line.split(",", -1);
            String cond = strs[4].trim().toLowerCase();
            if (cond.equals(""))
                cond = "na";
            conditions.put(strs[3].trim().toLowerCase(), cond);
        }
        br.close();
    }

    private void getConditionsFromCompactData(String dataPath, Multimap<String, String> conditions) throws Exception{
        BufferedReader br = FileIO.getReader(dataPath);
        String line = br.readLine(); // 1st line;
        String[] header = line.split(",", -1);
        int dimscount = header.length - 3;
        String[] dims = new String[dimscount];
        for (int i = 3; i < header.length; ++i)
            dims[i - 3] = header[i].trim().toLowerCase();
        while ((line = br.readLine()) != null) {
            String[] strs = line.split(",", -1);
            HashMap<String, String> ratingcontext = new HashMap<>();
            for (int i = 3; i < 3 + dimscount; ++i) {
                String cond = strs[i].trim().toLowerCase();
                if (cond.equals(""))
                    cond = "na";
                conditions.put(dims[i - 3], cond);
            }
        }
        br.close();
    }

    public String getHeader(Multimap<String, String> conditions){
        String header="User, Item, Rating";
        for(String dim:conditions.keySet())
        {
            for(String cond:conditions.get(dim))
                header+=", "+dim+":"+cond;
        }
        return header;
    }

    public String TransformationFromBinaryToBinary(String dataPath, boolean isTestSet, Multimap<String, String> conditions) throws Exception {
        // this method is only applied when the users manuall supply both train and test sets
        // it is because the format in train and test sets may not be consistent
        BufferedReader br = FileIO.getReader(dataPath);
        String line = br.readLine(); // 1st line;
        String[] header = line.split(",", -1);
        ArrayList<String> dims=new ArrayList<>();
        for(int i=3;i<header.length;++i){
           String strs[]= header[i].trim().split(":",-1);
           String dim=strs[0].trim().toLowerCase();
           if(!dims.contains(dim))
               dims.add(dim);
        }

        HashMap<String, HashMap<String, String>> newlines = new HashMap<>();
        if(conditions==null)
            conditions = LinkedHashMultimap.create(); // key=dim, value=cond, keep the order when we adding to it
        while ((line = br.readLine()) != null) {
            String[] strs = line.split(",", -1);
            HashMap<String, String> ratingcontext = new HashMap<>();
            for (int i = 3; i < header.length; ++i) {
                int cond = Integer.parseInt(strs[i].trim().toLowerCase());
                if(cond==0)
                    continue;
                else {
                    String dimcond=header[i];
                    String rs[]=dimcond.split(":",-1);
                    ratingcontext.put(rs[0].trim().toLowerCase(), rs[1].trim().toLowerCase());
                    if (!isTestSet)
                        conditions.put(rs[0].trim().toLowerCase(), rs[1].trim().toLowerCase());
                }
            }
            newlines.put(line, ratingcontext); // the whole line is key
        }
        br.close();

        String filename=(isTestSet)?"test.csv":"train.csv";

        this.PublishNewRatingFiles(outputfolder, conditions, newlines, false,filename);

        if (FileIO.exist(outputfolder + filename))
            return "Data transformaton completed (from Compact to Binary format). See new rating file: " + outputfolder + filename;
        return "Data transformaton completed (from Compact to Binary format). See " + outputfolder;
    }

    public String TransformationFromLooseToBinary(String dataPath, boolean isTestSet,Multimap<String, String> conditions) throws Exception {
        BufferedReader br = FileIO.getReader(dataPath);
        String line = br.readLine(); // 1st line;
        HashMap<String, HashMap<String, String>> newlines = new HashMap<>();
        if(conditions==null)
            conditions = LinkedHashMultimap.create(); // key=dim, value=cond, keep the order when we adding to it
        while ((line = br.readLine()) != null) {
            String[] strs = line.split(",", -1);
            String key = strs[0].trim().toLowerCase() + "," + strs[1].trim().toLowerCase() + "," + strs[2].trim().toLowerCase(); // key = user,item,rating
            String cond = strs[4].trim().toLowerCase();
            if (cond.equals(""))
                cond = "na";
            if(!isTestSet)
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

        String filename=(isTestSet)?"test.csv":"train.csv";

        this.PublishNewRatingFiles(outputfolder, conditions, newlines, true, filename);

        if (FileIO.exist(outputfolder + filename))
            return "Data transformaton completed (from Loose to Binary format). See new rating file: " + outputfolder + filename;
        else
            return "Data transformation failed. See output folder: " + outputfolder;
    }

    public String TransformationFromCompactToBinary(String dataPath, boolean isTestSet, Multimap<String, String> conditions) throws Exception {
        BufferedReader br = FileIO.getReader(dataPath);
        String line = br.readLine(); // 1st line;
        String[] header = line.split(",", -1);
        int dimscount = header.length - 3;
        String[] dims = new String[dimscount];
        for (int i = 3; i < header.length; ++i)
            dims[i - 3] = header[i].trim().toLowerCase();
        HashMap<String, HashMap<String, String>> newlines = new HashMap<>();
        if(conditions==null)
            conditions = LinkedHashMultimap.create(); // key=dim, value=cond, keep the order when we adding to it
        while ((line = br.readLine()) != null) {
            String[] strs = line.split(",", -1);
            HashMap<String, String> ratingcontext = new HashMap<>();
            for (int i = 3; i < 3 + dimscount; ++i) {
                String cond = strs[i].trim().toLowerCase();
                if (cond.equals(""))
                    cond = "na";
                ratingcontext.put(dims[i - 3], cond);
                if(!isTestSet)
                    conditions.put(dims[i - 3], cond);
            }
            newlines.put(line, ratingcontext); // the whole line is key
        }
        br.close();

        String filename=(isTestSet)?"test.csv":"train.csv";

        this.PublishNewRatingFiles(outputfolder, conditions, newlines, false,filename);

        if (FileIO.exist(outputfolder + filename))
            return "Data transformaton completed (from Compact to Binary format). See new rating file: " + outputfolder + filename;
        return "Data transformaton completed (from Compact to Binary format). See " + outputfolder;
    }

    private void PublishNewRatingFiles(String outputfolder, Multimap<String, String> conditions, HashMap<String, HashMap<String, String>> newlines, boolean isLoose, String filename) throws Exception {

        String header = this.getHeader(conditions);
        Logs.info(header);

        BufferedWriter bw = FileIO.getWriter(outputfolder + filename);
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
            if(flag_test == -1) {
                switch (flag_train) {
                    case 1: // it is binary format!
                        FileIO.copyFile(this.dataPath_train, this.outputfolder + "train.csv");
                        break;
                    case 2: // it is loose format!
                        Logs.warn("You rating data is in Loose format. CARSKit is working on transformation on the data format...");
                        Logs.info(this.TransformationFromLooseToBinary(this.dataPath_train,false, null));
                        break;
                    case 3: // it is compact format!
                        Logs.warn("You rating data is in Compact format. CARSKit is working on transformation on the data format...");
                        Logs.info(this.TransformationFromCompactToBinary(this.dataPath_train,false, null));
                        break;
                    default:
                        Logs.warn("You rating data is not shaped in the correct format. Please read our guideline on data preparation...");
                        break;
                }
            }else
            {
                // in this case, we need transferform both train and test sets

                // first of all, we need to collect unique information of dimension:condition by going through both train and test set
                // store them into the variable conditions
                Multimap<String, String> conditions=this.getConditions();
                // after that, we publish new files for train and test sets, even if they are already in binary format
                switch (flag_train) {
                    case 1: // it is binary format!
                        Logs.info(this.TransformationFromBinaryToBinary(this.dataPath_train,false, conditions));
                        break;
                    case 2: // it is loose format!
                        Logs.warn("You training data is in Loose format. CARSKit is working on transformation on the data format...");
                        Logs.info(this.TransformationFromLooseToBinary(this.dataPath_train,false, conditions));
                        break;
                    case 3: // it is compact format!
                        Logs.warn("You training data is in Compact format. CARSKit is working on transformation on the data format...");
                        Logs.info(this.TransformationFromCompactToBinary(this.dataPath_train,false, conditions));
                        break;
                    default:
                        Logs.warn("You training data is not shaped in the correct format. Please read our guideline on data preparation...");
                        break;
                }
                switch (flag_test) {
                    case 1: // it is binary format!
                        Logs.info(this.TransformationFromBinaryToBinary(this.dataPath_test,true, conditions));
                        break;
                    case 2: // it is loose format!
                        Logs.warn("You testing data is in Loose format. CARSKit is working on transformation on the data format...");
                        Logs.info(this.TransformationFromLooseToBinary(this.dataPath_test,true, conditions));
                        break;
                    case 3: // it is compact format!
                        Logs.warn("You testing data is in Compact format. CARSKit is working on transformation on the data format...");
                        Logs.info(this.TransformationFromCompactToBinary(this.dataPath_test,true, conditions));
                        break;
                    default:
                        Logs.warn("You testing data is not shaped in the correct format. Please read our guideline on data preparation...");
                        break;
                }

            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
