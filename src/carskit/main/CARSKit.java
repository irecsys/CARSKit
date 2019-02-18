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

package carskit.main;

import carskit.alg.baseline.avg.*;
import carskit.alg.baseline.cf.*;
import carskit.alg.baseline.ranking.*;
import carskit.alg.cars.adaptation.dependent.FM;
import carskit.alg.cars.adaptation.dependent.dev.*;
import carskit.alg.cars.adaptation.dependent.sim.*;
import carskit.alg.cars.adaptation.independent.CPTF;
import carskit.alg.cars.transformation.hybridfiltering.DCR;
import carskit.alg.cars.transformation.hybridfiltering.DCW;
import carskit.alg.cars.transformation.prefiltering.ExactFiltering;
import carskit.alg.cars.transformation.prefiltering.SPF;
import carskit.alg.cars.transformation.prefiltering.splitting.UserSplitting;
import com.google.common.collect.*;

import happy.coding.io.FileConfiger;
import happy.coding.io.FileIO;
import happy.coding.io.LineConfiger;
import happy.coding.io.Logs;
import happy.coding.io.Strings;
import happy.coding.math.Randoms;
import happy.coding.system.Dates;

import java.io.BufferedReader;
import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import carskit.data.processor.*;
import carskit.generic.Recommender;
import carskit.generic.IterativeRecommender;
import carskit.generic.Recommender.Measure;
import carskit.data.structure.SparseMatrix;
import carskit.alg.cars.transformation.prefiltering.splitting.*;


/**
 * carskit.main.Main Class of the CARSKit
 *
 * @author Yong Zheng
 *
 */

public class CARSKit {
    // version: MAJOR version (significant changes), followed by MINOR version (small changes, bug fixes)
    protected static String version = "0.3.5";
    protected static String defaultConfigFileName = "setting.conf";
    // is only to print measurements
    public static boolean isMeasuresOnly = false;
    // output directory path
    protected static String WorkingFolder;
    protected static String DefaultWorkingFolder = "CARSKit.Workspace";
    protected static String WorkingPath;

    // setting
    protected FileConfiger cf;
    protected List<String> configFiles;
    protected String algorithm;

    protected float binThold;
    private boolean fullStat = false;

    // DAO
    protected DataDAO rateDao;
    protected SparseMatrix rateMatrix; // the shape of this matrix depends on which algorithm will be used
    protected LineConfiger ratingOptions, outputOptions;

    public static void main(String[] args) throws Exception{

        try {
            new CARSKit().execute(args);

        } catch (Exception e) {
            // capture exception to log file
            Logs.error(e.getMessage());

            e.printStackTrace();
        }
    }


    /**
     * run the library
     */
    protected void execute(String[] args) throws Exception {
        // process librec arguments
        cmdLine(args);

        // multiple runs at one time
        for (String config : configFiles) {

            // reset general settings
            preset(config);

            // prepare data
            readData();

            // run a specific algorithm
            runAlgorithm();
        }

        // collect results
        String filename = (configFiles.size() > 1 ? "multiAlgorithms" : algorithm) + "@" + Dates.now() + ".txt";
        String results = Recommender.workingPath + filename;
        FileIO.copyFile("results.txt", results);


    }

    /**
     * reset general (and static) settings
     */
    protected void preset(String configFile) throws Exception {

        // a new configer
        cf = new FileConfiger(configFile);
        String separator=System.getProperty("file.separator");

        // seeding the general recommender
        Recommender.cf = cf;

        // reset recommenders' static properties
        Recommender.resetStatics = true;
        IterativeRecommender.resetStatics = true;

        // inputs
        String currentRatingFile=cf.getPath("dataset.ratings");
        if(!FileIO.exist(currentRatingFile))
            Logs.error("Your rating file path is incorrect: File doesn't exist. Please double check your configuration.");
        else {
            String currentFilePath = currentRatingFile.substring(0, currentRatingFile.lastIndexOf(separator) + 1);

            // outputs
            // there are more output options in the configuration
            outputOptions = cf.getParamOptions("output.setup");
            if (outputOptions != null) {
                isMeasuresOnly = outputOptions.contains("--measures-only");
            }
            WorkingFolder = outputOptions.getString("-folder", DefaultWorkingFolder);
            // make output directory
            WorkingPath = currentFilePath + WorkingFolder + separator;
            Logs.info("WorkingPath: "+WorkingPath);
            Recommender.workingPath = FileIO.makeDirectory(WorkingPath);
        }
        // initialize random seed
        LineConfiger evalOptions = cf.getParamOptions("evaluation.setup");
        Randoms.seed(evalOptions.getLong("--rand-seed", System.currentTimeMillis())); // initial random seed
    }

    protected boolean isBinaryNumber(int number) { int copyOfInput = number; while (copyOfInput != 0) { if (copyOfInput % 10 > 1) { return false; } copyOfInput = copyOfInput / 10; } return true; }

    protected int validateDataFormat(String dataPath) throws Exception {

        // flag=0, did not check; flag=1, binary format; flag=2, loose format; flag=3, compact format
        int flag=0;
        BufferedReader br = FileIO.getReader(dataPath);
        String header = br.readLine(); // 1st line;
        String dataline = br.readLine();
        br.close();

        String[] sheader=header.split(",",-1);
        String[] sdata=dataline.split(",",-1);
        String lastColumn=sheader[sheader.length-1].trim().toLowerCase();
        if(sheader[sheader.length-2].trim().toLowerCase().equals("dimension") &&
                lastColumn.equals("condition"))
        {
            flag=2; // it is loose format
        }else
        {
            boolean isBinary=true; // header is as "dimension:condition"
            for(int i=3;i<sheader.length;++i)
            {
                if(sheader[i].indexOf(":")==-1 || isBinaryNumber(Integer.valueOf(sdata[i]))==false)
                {
                    isBinary=false;
                    break;
                }
            }
            if(isBinary) // basically, it is probably binary format
            {
                flag=1; // it is binary format
            }else
            {
                flag=3; // it is compact format
            }
        }
        return flag; // those are very basic validations; if there are data errors, the readData() will throw errors
    }

    /**
     * read input data
     */
    protected void readData() throws Exception {
        // DAO object

        String OriginalRatingDataPath=cf.getPath("dataset.ratings");
        Logs.info("Your original rating data path: "+OriginalRatingDataPath);
        Logs.info("Current working path: "+WorkingPath);

        if(!FileIO.exist(OriginalRatingDataPath))
            Logs.error("Your rating file path is incorrect: File doesn't exist. Please double check your configuration.");

        // data configuration
        ratingOptions = cf.getParamOptions("ratings.setup");
        int dataTransformation = ratingOptions.getInt("-datatransformation");
        if(dataTransformation>0) {

            // check whether the evaluation method is the way of train-test validation by manually supplying train and test sets
            // If yes, we may need to transferform both the training and testing set
            String setup = cf.getString("evaluation.setup");
            LineConfiger evalOptions = new LineConfiger(setup);
            String evMethod = evalOptions.getMainParam().toLowerCase().trim();
            boolean testTransfer=false;
            String testPath=null;
            if(evMethod.equals("test-set"))
            {
                testTransfer = true;
                testPath = evalOptions.getString("-f");
            }

            DataTransformer transformer = new DataTransformer();
            int flag_train = validateDataFormat(OriginalRatingDataPath);
            int flag_test = -1;
            if(testTransfer)
                flag_test = validateDataFormat(testPath);
            transformer.setParameters(flag_train, OriginalRatingDataPath, flag_test, testPath, WorkingPath);
            Thread t = new Thread(transformer);
            t.start();
            t.join(); // for large data, the transformation and output to external file may take time!
        }

        rateDao = new DataDAO(WorkingPath+"train.csv");

        // rating threshold
        binThold = ratingOptions.getFloat("-threshold");
        // print full stat?
        fullStat = (ratingOptions.getInt("-fullstat",-1) > 0)?true:false;
        rateDao.setFullStat(fullStat);

        rateMatrix = rateDao.readData(binThold);
        rateDao.printSpecs();

        Recommender.rateMatrix = rateMatrix;
        Recommender.rateDao = rateDao;
        Recommender.binThold = binThold;
    }


    /**
     * process arguments specified at the command line
     *
     * @param args
     *            command line arguments
     */
    protected void cmdLine(String[] args) throws Exception {

        if (args == null || args.length < 1) {
            if (configFiles == null)
                configFiles = Arrays.asList(defaultConfigFileName);
            return;
        }

        LineConfiger paramOptions = new LineConfiger(args);
        configFiles = paramOptions.contains("-c") ? paramOptions.getOptions("-c") : Arrays.asList(defaultConfigFileName);
        if(paramOptions.contains("-version")){
            about();
            System.exit(0);
        }
        if (paramOptions.contains("-v")) {
            // print out short version information
            System.out.println("CARSKit version " + version);
            System.exit(0);
        }
    }


    /**
     * write a matrix data into a file
     */



    protected void runAlgorithm() throws Exception {

        // evaluation setup
        String setup = cf.getString("evaluation.setup");
        LineConfiger evalOptions = new LineConfiger(setup);

        // debug information
        if (isMeasuresOnly)
            Logs.debug("With Setup: {}", setup);
        else
            Logs.info("With Setup: {}", setup);

        Recommender algo = null;

        DataSplitter ds = new DataSplitter(rateMatrix);
        SparseMatrix[] data = null;

        int N;
        double ratio;

        switch (evalOptions.getMainParam().toLowerCase()) {
            case "cv": // run by cross validation
                runCrossValidation(evalOptions);
                return;
            case "test-set":
                DataDAO testDao = new DataDAO(WorkingPath+"test.csv", rateDao.getUserIds(), rateDao.getItemIds(), rateDao.getContextIds(), rateDao.getUserItemIds(),
                        rateDao.getContextDimensionIds(), rateDao.getContextConditionIds(), rateDao.getURatedList(), rateDao.getIRatedList(),
                        rateDao.getDimConditionsList(), rateDao.getConditionDimensionMap(), rateDao.getConditionContextsList(), rateDao.getContextConditionsList(),
                        rateDao.getUiUserIds(), rateDao.getUiItemIds());

                SparseMatrix testData = testDao.readData(binThold);
                data = new SparseMatrix[] { rateMatrix, testData};
                break;
            case "given-ratio":
                ratio = evalOptions.getDouble("-r", 0.8);
                data = ds.getRatioByRating(ratio);
                break;
            default:
                ratio = evalOptions.getDouble("-r", 0.8);
                data = ds.getRatioByRating(ratio);
                break;
        }

        algo = getRecommender(data, -1);
        algo.execute();

        printEvalInfo(algo, algo.measures);
    }

    /**
     * print out the evaluation information for a specific algorithm
     */
    private void printEvalInfo(Recommender algo, Map<Measure, Double> ms) throws Exception {

        String result = Recommender.getEvalInfo(ms);
        // we add quota symbol to indicate the textual format of time
        String time = String.format("'%s','%s'", Dates.parse(ms.get(Measure.TrainTime).longValue()),
                Dates.parse(ms.get(Measure.TestTime).longValue()));
        // double commas as the separation of results and configuration
        String evalInfo = String.format("Final Results by %s, %s, %s, Time: %s%s", algo.algoName, result, algo.toString(), time,
                (outputOptions.contains("--measures-only") ? "" : "\n"));

        Logs.info(evalInfo);

        // copy to clipboard for convenience, useful for a single run
        if (outputOptions.contains("--to-clipboard")) {
            Strings.toClipboard(evalInfo);
            Logs.debug("Have been copied to clipboard!");
        }

        // append to a specific file, useful for multiple runs
        if (outputOptions.contains("--to-file")) {
            String filePath = outputOptions.getString("--to-file", WorkingPath + algorithm + ".txt");
            FileIO.writeString(filePath, evalInfo, true);
            Logs.debug("Have been collected to file: {}", filePath);
        }
    }

    private void runCrossValidation(LineConfiger params) throws Exception {

        int kFold = params.getInt("-k", 5);
        boolean isParallelFold = params.isOn("-p", true);

        DataSplitter ds = new DataSplitter(rateMatrix, kFold);

        Thread[] ts = new Thread[kFold];
        Recommender[] algos = new Recommender[kFold];

        for (int i = 0; i < kFold; i++) {
            Recommender algo = getRecommender(ds.getKthFold(i + 1), i + 1);

            algos[i] = algo;
            ts[i] = new Thread(algo);
            ts[i].start();

            if (!isParallelFold)
                ts[i].join();
        }


        if (isParallelFold)
            for (Thread t : ts)
                t.join();

        // average performance of k-fold
        Map<Measure, Double> avgMeasure = new HashMap<>();
        for (Recommender algo : algos) {
            //Logs.info("Measures: "+algo.measures.entrySet().size());
            for (Entry<Measure, Double> en : algo.measures.entrySet()) {
                Measure m = en.getKey();
                double val = avgMeasure.containsKey(m) ? avgMeasure.get(m) : 0.0;
                avgMeasure.put(m, val + en.getValue() / kFold);
            }
        }

        printEvalInfo(algos[0], avgMeasure);
    }


    protected Recommender getRecommender(SparseMatrix[] data, int fold) throws Exception {

        algorithm = cf.getString("recommender");
        LineConfiger algOptions = new LineConfiger(algorithm);

        SparseMatrix trainMatrix = data[0], testMatrix = data[1];

        // output data
        //writeData(trainMatrix, testMatrix, fold);

        switch (algOptions.getMainParam().toLowerCase()) {

            case "globalavg":
                return new GlobalAverage(trainMatrix, testMatrix, fold);
            case "useravg":
                return new UserAverage(trainMatrix, testMatrix, fold);
            case "itemavg":
                return new ItemAverage(trainMatrix, testMatrix, fold);
            case "contextavg":
                return new ContextAverage(trainMatrix, testMatrix, fold);
            case "useritemavg":
                return new UserItemAverage(trainMatrix, testMatrix, fold);
            case "usercontextavg":
                return new UserContextAverage(trainMatrix, testMatrix, fold);
            case "itemcontextavg":
                return new ItemContextAverage(trainMatrix, testMatrix, fold);
            case "itemknn":
                return new ItemKNN(trainMatrix, testMatrix, fold);
            case "userknn":
                return new UserKNN(trainMatrix, testMatrix, fold);
            case "slopeone":
                return new SlopeOne(trainMatrix, testMatrix, fold);
            case "biasedmf":
                return new BiasedMF(trainMatrix, testMatrix, fold);
            case "pmf":
                return new PMF(trainMatrix, testMatrix, fold);
            case "bpmf":
                return new BPMF(trainMatrix, testMatrix, fold);
            case "nmf":
                return new NMF(trainMatrix, testMatrix, fold);
            case "svd++":
                return new SVDPlusPlus(trainMatrix, testMatrix, fold);
            case "slim":
                return new SLIM(trainMatrix, testMatrix, fold);
            case "bpr":
                return new BPR(trainMatrix, testMatrix, fold);
            case "lrmf":
                return new LRMF(trainMatrix, testMatrix, fold);
            case "rankals":
                return new RankALS(trainMatrix, testMatrix, fold);
            case "ranksgd":
                return new RankSGD(trainMatrix, testMatrix, fold);
            case "usersplitting":
            {
                String recsys_traditional=algOptions.getString("-traditional").trim().toLowerCase();
                int minListLenU=algOptions.getInt("-minlenu", 2);
                UserSplitting usp=new UserSplitting(rateDao.numUsers(),rateDao.getConditionContextsList(), rateDao.getURatedList());
                Table<Integer, Integer, Integer> userIdMapper=usp.split(trainMatrix, minListLenU);
                Logs.info("User Splitting is done... Algorithm '"+recsys_traditional+"' will be applied to the transformed data set.");
                Recommender recsys=null;
                switch(recsys_traditional)
                {
                    ///////// baseline algorithms //////////////////////////////////////////////////////////
                    case "globalavg":
                        recsys=new GlobalAverage(trainMatrix, testMatrix, fold);break;
                    case "useravg":
                        recsys=new UserAverage(trainMatrix, testMatrix, fold);break;
                    case "itemavg":
                        recsys=new ItemAverage(trainMatrix, testMatrix, fold);break;
                    case "contextavg":
                        recsys=new ContextAverage(trainMatrix, testMatrix, fold);break;
                    case "useritemavg":
                        recsys=new UserItemAverage(trainMatrix, testMatrix, fold);break;
                    case "usercontextavg":
                        recsys=new UserContextAverage(trainMatrix, testMatrix, fold);break;
                    case "itemcontextavg":
                        recsys=new ItemContextAverage(trainMatrix, testMatrix, fold);break;
                    case "itemknn":
                        recsys=new ItemKNN(trainMatrix, testMatrix, fold);break;
                    case "userknn":
                        recsys=new UserKNN(trainMatrix, testMatrix, fold);break;
                    case "slopeone":
                        recsys=new SlopeOne(trainMatrix, testMatrix, fold);break;
                    case "biasedmf":
                        recsys=new BiasedMF(trainMatrix, testMatrix, fold);break;
                    case "pmf":
                        recsys=new PMF(trainMatrix, testMatrix, fold);break;
                    case "bpmf":
                        recsys=new BPMF(trainMatrix, testMatrix, fold); break;
                    case "nmf":
                        recsys=new NMF(trainMatrix, testMatrix, fold); break;
                    case "svd++":
                        recsys=new SVDPlusPlus(trainMatrix, testMatrix, fold); break;
                    case "slim":
                        recsys=new SLIM(trainMatrix, testMatrix, fold);break;
                    case "bpr":
                        recsys=new BPR(trainMatrix, testMatrix, fold);break;
                    case "lrmf":
                        recsys=new LRMF(trainMatrix, testMatrix, fold);break;
                    case "rankals":
                        recsys=new RankALS(trainMatrix, testMatrix, fold);break;
                    case "ranksgd":
                        recsys=new RankSGD(trainMatrix, testMatrix, fold);break;
                    default:
                        recsys=null;
                }
                if(recsys==null)
                    throw new Exception("No recommender is specified!");
                else
                {
                    recsys.setIdMappers(userIdMapper,null);
                    return recsys;
                }
            }

            ///////// Context-aware Splitting algorithms //////////////////////////////////////////////////////////
            case "itemsplitting":
            {
                String recsys_traditional=algOptions.getString("-traditional").trim().toLowerCase();
                int minListLenI=algOptions.getInt("-minleni", 2);
                ItemSplitting isp=new ItemSplitting(rateDao.numItems(),rateDao.getConditionContextsList(), rateDao.getIRatedList());
                Table<Integer, Integer, Integer> itemIdMapper=isp.split(trainMatrix, minListLenI);
                Logs.info("Item Splitting is done... Algorithm '"+recsys_traditional+"' will be applied to the transformed data set.");
                Recommender recsys=null;
                switch(recsys_traditional)
                {

                    case "globalavg":
                        recsys=new GlobalAverage(trainMatrix, testMatrix, fold);break;
                    case "useravg":
                        recsys=new UserAverage(trainMatrix, testMatrix, fold);break;
                    case "itemavg":
                        recsys=new ItemAverage(trainMatrix, testMatrix, fold);break;
                    case "contextavg":
                        recsys=new ContextAverage(trainMatrix, testMatrix, fold);break;
                    case "useritemavg":
                        recsys=new UserItemAverage(trainMatrix, testMatrix, fold);break;
                    case "usercontextavg":
                        recsys=new UserContextAverage(trainMatrix, testMatrix, fold);break;
                    case "itemcontextavg":
                        recsys=new ItemContextAverage(trainMatrix, testMatrix, fold);break;
                    case "itemknn":
                        recsys=new ItemKNN(trainMatrix, testMatrix, fold);break;
                    case "userknn":
                        recsys=new UserKNN(trainMatrix, testMatrix, fold);break;
                    case "slopeone":
                        recsys=new SlopeOne(trainMatrix, testMatrix, fold);break;
                    case "biasedmf":
                        recsys=new BiasedMF(trainMatrix, testMatrix, fold);break;
                    case "pmf":
                        recsys=new PMF(trainMatrix, testMatrix, fold);break;
                    case "bpmf":
                        recsys=new BPMF(trainMatrix, testMatrix, fold); break;
                    case "nmf":
                        recsys=new NMF(trainMatrix, testMatrix, fold); break;
                    case "svd++":
                        recsys=new SVDPlusPlus(trainMatrix, testMatrix, fold); break;
                    case "slim":
                        recsys=new SLIM(trainMatrix, testMatrix, fold);break;
                    case "bpr":
                        recsys=new BPR(trainMatrix, testMatrix, fold);break;
                    case "lrmf":
                        recsys=new LRMF(trainMatrix, testMatrix, fold);break;
                    case "rankals":
                        recsys=new RankALS(trainMatrix, testMatrix, fold);break;
                    case "ranksgd":
                        recsys=new RankSGD(trainMatrix, testMatrix, fold);break;
                    default:
                        recsys=null;
                }
                if(recsys==null)
                    throw new Exception("No recommender is specified!");
                else
                {
                    recsys.setIdMappers(null, itemIdMapper);
                    return recsys;
                }
            }
            case "uisplitting":
            {
                String recsys_traditional=algOptions.getString("-traditional").trim().toLowerCase();
                int minListLenU=algOptions.getInt("-minlenu", 2);
                int minListLenI=algOptions.getInt("-minleni", 2);
                UISplitting sp=new UISplitting(rateDao.numUsers(), rateDao.numItems(), rateDao.getConditionContextsList(), rateDao.getURatedList(), rateDao.getIRatedList());
                Table<Integer, Integer, Integer> itemIdMapper=sp.splitItem(trainMatrix, minListLenI);
                Table<Integer, Integer, Integer> userIdMapper=sp.splitUser(trainMatrix, minListLenU);
                Logs.info("UI Splitting is done... Algorithm '"+recsys_traditional+"' will be applied to the transformed data set.");
                Recommender recsys=null;
                switch(recsys_traditional)
                {

                    case "globalavg":
                        recsys=new GlobalAverage(trainMatrix, testMatrix, fold);break;
                    case "useravg":
                        recsys=new UserAverage(trainMatrix, testMatrix, fold);break;
                    case "itemavg":
                        recsys=new ItemAverage(trainMatrix, testMatrix, fold);break;
                    case "contextavg":
                        recsys=new ContextAverage(trainMatrix, testMatrix, fold);break;
                    case "useritemavg":
                        recsys=new UserItemAverage(trainMatrix, testMatrix, fold);break;
                    case "usercontextavg":
                        recsys=new UserContextAverage(trainMatrix, testMatrix, fold);break;
                    case "itemcontextavg":
                        recsys=new ItemContextAverage(trainMatrix, testMatrix, fold);break;
                    case "itemknn":
                        recsys=new ItemKNN(trainMatrix, testMatrix, fold);break;
                    case "userknn":
                        recsys=new UserKNN(trainMatrix, testMatrix, fold);break;
                    case "slopeone":
                        recsys=new SlopeOne(trainMatrix, testMatrix, fold);break;
                    case "biasedmf":
                        recsys=new BiasedMF(trainMatrix, testMatrix, fold);break;
                    case "pmf":
                        recsys=new PMF(trainMatrix, testMatrix, fold);break;
                    case "bpmf":
                        recsys=new BPMF(trainMatrix, testMatrix, fold); break;
                    case "nmf":
                        recsys=new NMF(trainMatrix, testMatrix, fold); break;
                    case "svd++":
                        recsys=new SVDPlusPlus(trainMatrix, testMatrix, fold); break;
                    case "slim":
                        recsys=new SLIM(trainMatrix, testMatrix, fold);break;
                    case "bpr":
                        recsys=new BPR(trainMatrix, testMatrix, fold);break;
                    case "lrmf":
                        recsys=new LRMF(trainMatrix, testMatrix, fold);break;
                    case "rankals":
                        recsys=new RankALS(trainMatrix, testMatrix, fold);break;
                    case "ranksgd":
                        recsys=new RankSGD(trainMatrix, testMatrix, fold);break;
                    default:
                        recsys=null;
                }
                if(recsys==null)
                    throw new Exception("No recommender is specified!");
                else
                {
                    recsys.setIdMappers(userIdMapper, itemIdMapper);
                    return recsys;
                }
            }

            case "exactfiltering":
            {
                return new ExactFiltering(trainMatrix, testMatrix, fold);
            }

            case "dcr":
            {
                return new DCR(trainMatrix, testMatrix, fold);
            }

            case "dcw":
            {
                return new DCW(trainMatrix, testMatrix, fold);
            }

            case "spf":
            {
                return new SPF(trainMatrix, testMatrix, fold);
            }

            ///////// Context-aware recommender: Tensor Factorization //////////////////////////////////////////////////////////
            case "cptf":
            {
                rateDao.LoadAsTensor();
                return new CPTF(trainMatrix, testMatrix, fold);
            }

            ///////// Context-aware recommender: CAMF //////////////////////////////////////////////////////////
            case "camf_c":
                return new CAMF_C(trainMatrix, testMatrix, fold);
            case "camf_ci":
                return new CAMF_CI(trainMatrix, testMatrix, fold);
            case "camf_cu":
                return new CAMF_CU(trainMatrix, testMatrix, fold);
            case "camf_cuci":
                return new CAMF_CUCI(trainMatrix, testMatrix, fold);
            case "camf_ics":
                return new CAMF_ICS(trainMatrix, testMatrix, fold);
            case "camf_lcs":
                return new CAMF_LCS(trainMatrix, testMatrix, fold);
            case "camf_mcs":
                return new CAMF_MCS(trainMatrix, testMatrix, fold);


            ///////// Context-aware recommender: CSLIM //////////////////////////////////////////////////////////
            case "cslim_c":
                return new CSLIM_C(trainMatrix, testMatrix, fold);
            case "cslim_cu":
                return new CSLIM_CU(trainMatrix, testMatrix, fold);
            case "cslim_ci":
                return new CSLIM_CI(trainMatrix, testMatrix, fold);
            case "cslim_cuci":
                return new CSLIM_CUCI(trainMatrix, testMatrix, fold);
            case "gcslim_cc":
                return new GCSLIM_CC(trainMatrix, testMatrix, fold);
            case "cslim_ics":
                return new CSLIM_ICS(trainMatrix, testMatrix, fold);
            case "cslim_lcs":
                return new CSLIM_LCS(trainMatrix, testMatrix, fold);
            case "cslim_mcs":
                return new CSLIM_MCS(trainMatrix, testMatrix, fold);
            case "gcslim_ics":
                return new GCSLIM_ICS(trainMatrix, testMatrix, fold);
            case "gcslim_lcs":
                return new GCSLIM_LCS(trainMatrix, testMatrix, fold);
            case "gcslim_mcs":
                return new GCSLIM_MCS(trainMatrix, testMatrix, fold);


            ////////////// Other context-aware recommendation algorithms /////////////////////////////////////////
            case "fm":
                return new FM(trainMatrix, testMatrix, fold);


            default:
                throw new Exception("No recommender is specified!");
        }
    }




    /**
     * set the configuration file to be used
     */
    public void setConfigFiles(String... configurations) {
        configFiles = Arrays.asList(configurations);
    }

    /**
     * Print out software information
     */
    private void about() {
        String about = "\nCARSKit version " + version + ", copyright (C) 2015-2019 Yong Zheng \n\n"

		        /* Description */
                + "CARSKit is free software: you can redistribute it and/or modify \n"
                + "it under the terms of the GNU General Public License as published by \n"
                + "the Free Software Foundation, either version 3 of the License, \n"
                + "or (at your option) any later version. \n\n"

				/* Usage */
                + "CARSKit is distributed in the hope that it will be useful, \n"
                + "but WITHOUT ANY WARRANTY; without even the implied warranty of \n"
                + "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the \n"
                + "GNU General Public License for more details. \n\n"

				/* licence */
                + "You should have received a copy of the GNU General Public License \n"
                + "along with CARSKit. If not, see <http://www.gnu.org/licenses/>.";

        System.out.println(about);
    }

}
